#ifdef FMOE_USE_NCCL

#include <cstdlib>
#include <vector>
#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>

#include "fused_compute.cuh"

long pipeline_gran = -1;

std::vector<torch::Tensor> _fused_forward(
        torch::Tensor input_buf,
        torch::Tensor weight1,
        torch::Tensor weight2,
        torch::Tensor local_expert_count,
        torch::Tensor global_expert_count,
        long global_batch_size,
        long n_workers, bool has_bias) {

    if (pipeline_gran == -1) {
        char* p = getenv("FMOE_FUSE_GRAN");
        if (p) {
            pipeline_gran = atoi(p);
        } else {
            pipeline_gran = 4;
        }
    }

    const auto num_expert = local_expert_count.size(0) / n_workers;
    const auto d_hidden = weight1.size(1);
    const auto d_model = weight1.size(2);

    auto smgr = getCudaStreamManager(input_buf.device().index());

    auto global_input_buf = input_buf.new_empty({global_batch_size, d_model});
    auto global_middle_buf = input_buf.new_empty({global_batch_size, d_hidden});
    auto global_output_buf = input_buf.new_empty({global_batch_size, d_model});
    auto output_buf = input_buf.new_empty({input_buf.size(0), d_model});

    int rank;
    ncclCommUserRank(smgr->ncclcomm, &rank);
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input_buf.scalar_type(), 
            "fmoe_cuda_fused_forward", ([&] {
        fmoe_cuda_fused_forward_impl(
            input_buf.data_ptr<scalar_t>(),
            weight1.data_ptr<scalar_t>(),
            weight2.data_ptr<scalar_t>(),

            global_input_buf.data_ptr<scalar_t>(),
            global_middle_buf.data_ptr<scalar_t>(),
            global_output_buf.data_ptr<scalar_t>(),
            output_buf.data_ptr<scalar_t>(),

            local_expert_count.data_ptr<long>(),
            global_expert_count.data_ptr<long>(),
            d_model, d_hidden, num_expert, rank, n_workers, has_bias,
            pipeline_gran, smgr);
    }));
    return {output_buf, global_input_buf, global_middle_buf, global_output_buf};
}

std::vector<torch::Tensor> _fused_backward(
        torch::Tensor input_buf,
        torch::Tensor weight1,
        torch::Tensor weight2,
        torch::Tensor middle_buf,
        torch::Tensor output_buf,
        torch::Tensor grad_out,
        torch::Tensor local_expert_count,
        torch::Tensor global_expert_count,
        long global_batch_size,
        long n_workers, bool has_bias) {
    const auto num_expert = local_expert_count.size(0) / n_workers;
    const auto d_hidden = weight1.size(1);
    const auto d_model = weight1.size(2);

    auto smgr = getCudaStreamManager(input_buf.device().index());

    auto global_grad_out = input_buf.new_empty({global_batch_size, d_model});
    auto grad_middle = input_buf.new_empty({global_batch_size, d_hidden});
    auto global_grad_in = input_buf.new_empty({global_batch_size, d_model});

    auto grad_in = input_buf.new_empty({input_buf.size(0), d_model});
    auto grad_weight1 = torch::empty_like(weight1);
    auto grad_weight2 = torch::empty_like(weight2);

    int rank;
    ncclCommUserRank(smgr->ncclcomm, &rank);
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input_buf.scalar_type(), 
            "fmoe_cuda_fused_backward", ([&] {
        fmoe_cuda_fused_backward_impl(
            input_buf.data_ptr<scalar_t>(),
            weight1.data_ptr<scalar_t>(),
            weight2.data_ptr<scalar_t>(),
            middle_buf.data_ptr<scalar_t>(),
            output_buf.data_ptr<scalar_t>(),
            grad_out.data_ptr<scalar_t>(),

            global_grad_out.data_ptr<scalar_t>(),
            global_grad_in.data_ptr<scalar_t>(),

            grad_middle.data_ptr<scalar_t>(),
            grad_weight1.data_ptr<scalar_t>(),
            grad_weight2.data_ptr<scalar_t>(),
            grad_in.data_ptr<scalar_t>(),

            local_expert_count.data_ptr<long>(),
            global_expert_count.data_ptr<long>(),
            d_model, d_hidden, num_expert, rank, n_workers, has_bias,
            pipeline_gran, smgr);
    }));
    return {grad_in, grad_weight1, grad_weight2};
}

#endif

