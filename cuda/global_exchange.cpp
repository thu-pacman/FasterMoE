#include "global_exchange.h"
#include "utils/fmoe_utils.h"
#include <torch/extension.h>

#ifdef FMOE_USE_NCCL
#include <nccl.h>

torch::Tensor _expert_exchange(
        torch::Tensor local_expert_count,
        long n_expert, long n_workers) {
    auto global_expert_count = torch::empty_like(local_expert_count);
    auto smgr = getCudaStreamManager(local_expert_count.device().index());

    fmoe_cuda_expert_exchange_impl(
            local_expert_count.data_ptr<long>(),
            global_expert_count.data_ptr<long>(),
            n_expert, n_workers,
            smgr);
    return global_expert_count;
}

torch::Tensor _global_scatter(
        torch::Tensor input_buf,
        torch::Tensor local_expert_count,
        torch::Tensor global_expert_count,
        long batch_size, long n_workers) {
    CHECK_INPUT(input_buf);

    auto n_expert = local_expert_count.size(0) / n_workers;
    auto in_feat = input_buf.size(1);
    auto global_input_buf = input_buf.new_empty({batch_size, in_feat});
    auto smgr = getCudaStreamManager(input_buf.device().index());

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input_buf.scalar_type(), 
            "fmoe_cuda_global_scatter", ([&] {
        fmoe_cuda_global_scatter_impl<scalar_t>(
            input_buf.data_ptr<scalar_t>(),
            local_expert_count.data_ptr<long>(),
            global_expert_count.data_ptr<long>(),
            global_input_buf.data_ptr<scalar_t>(),
            in_feat, n_expert, n_workers,
            smgr
        );
    }));
    return global_input_buf;
}

torch::Tensor _global_gather(
        torch::Tensor output_buf,
        torch::Tensor local_expert_count,
        torch::Tensor global_expert_count,
        long batch_size, long n_workers) {
    CHECK_INPUT(output_buf);

    auto n_expert = local_expert_count.size(0) / n_workers;
    auto out_feat = output_buf.size(1);
    auto local_output_buf = output_buf.new_empty({batch_size, out_feat});
    auto smgr = getCudaStreamManager(output_buf.device().index());

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(output_buf.scalar_type(), 
            "fmoe_cuda_global_gather", ([&] {
        fmoe_cuda_global_gather_impl<scalar_t>(
            output_buf.data_ptr<scalar_t>(),
            local_expert_count.data_ptr<long>(),
            global_expert_count.data_ptr<long>(),
            local_output_buf.data_ptr<scalar_t>(),
            out_feat, n_expert, n_workers,
            smgr
        );
    }));
    return local_output_buf;
}

#include <c10d/ProcessGroupNCCL.hpp>

class HackNCCLGroup: public c10d::ProcessGroupNCCL {
public:
    ncclComm_t getcomm(at::Device dev) {
        ncclUniqueId ncclID;
        int rank = getRank();
        if (rank == 0) {
            ncclGetUniqueId(&ncclID);
        }
#if defined(TORCH_VERSION_MAJOR) && (TORCH_VERSION_MAJOR > 1 || \
        (TORCH_VERSION_MAJOR == 1 && TORCH_VERSION_MINOR >= 8))
        broadcastUniqueNCCLID(&ncclID,
                c10d::OpType::SEND,
                "fastmoe_nccl_comm",
                rank);
#else
        broadcastUniqueNCCLID(&ncclID);
#endif
        ncclComm_t comm;
        NCCL_SAFE_CALL(ncclCommInitRank(&comm, getSize(), ncclID, rank));
        return comm;
    }
};

void _ensure_nccl(c10d::ProcessGroupNCCL& p, torch::Tensor t) {
    auto smgr = getCudaStreamManager(t.device().index());
    if (smgr->ncclgood) {
        return;
    }
    HackNCCLGroup* h = (HackNCCLGroup*)(void*)&p;
    smgr->ncclcomm = h->getcomm(t.device());
    if (smgr->ncclcomm != 0) {
        smgr->ncclgood = 1;
    } else {
        std::cerr << "Nccl initialization failed\n";
    }
}

torch::Tensor _exchange_cache_info(
        torch::Tensor sent_models,
        long num_expert,
        long world_size) {
    
    CHECK_INPUT(sent_models);

    auto smgr = getCudaStreamManager(sent_models.device().index());

    torch::Tensor stored_models = sent_models.new_empty({world_size, num_expert});

    fmoe_cuda_exchange_cache_info_impl(sent_models.data_ptr<bool>(), stored_models.data_ptr<bool>(), num_expert, world_size, smgr);

    return stored_models;
}

void _model_exchange(
        torch::Tensor sent_models,
        torch::Tensor stored_models,
        std::vector<std::vector<torch::Tensor>> local_params,
        std::vector<std::vector<std::vector<torch::Tensor>>> params,
        long num_expert, long world_size, bool fused) {

    for (int j = 0; j < num_expert; j++) {
        for (auto t : local_params[j]) {
            CHECK_INPUT(t);
        }

        for (int i = 0; i < world_size; i++){
            if (params[i].size() <= 0) continue;
            for (auto t : params[i][j]) {
                CHECK_INPUT(t);
            }
        }
    }

    auto smgr = getCudaStreamManager(local_params[0][0].device().index());

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(local_params[0][0].scalar_type(), 
            "fmoe_cuda_model_exchange", ([&] {
        fmoe_cuda_model_exchange_impl<scalar_t>(
            sent_models.data_ptr<bool>(),
            stored_models.data_ptr<bool>(),
            local_params,
            params,
            num_expert, world_size, fused, // TODO should fused be here
            smgr
        );
    }));

}

#endif  // FMOE_USE_NCCL
