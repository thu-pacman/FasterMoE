#include "stream_manager.h"
#ifdef FMOE_USE_NCCL
#include <vector>
#include <torch/extension.h>

void fmoe_cuda_expert_exchange_impl(
        const long* local_expert_count, 
        long* global_expert_count, 
        int n_expert, int world_size,
        CudaStreamManager* smgr) {
    NCCL_SAFE_CALL(ncclGroupStart());
    for (int i = 0; i < world_size; ++i) {
        NCCL_SAFE_CALL(ncclSend(
                local_expert_count + n_expert * i,
                n_expert,
                ncclInt64,
                i,
                smgr->ncclcomm,
                smgr->stream(0)));
        NCCL_SAFE_CALL(ncclRecv(
                global_expert_count + n_expert * i,
                n_expert,
                ncclInt64,
                i,
                smgr->ncclcomm,
                smgr->stream(0)));
    }
    NCCL_SAFE_CALL(ncclGroupEnd());
    smgr->sync(1);
}

template<typename scalar_t>
void fmoe_cuda_global_scatter_impl(
    const scalar_t* local_input_buf,
    const long* local_expert_count,
    const long* global_expert_count,
    scalar_t* input_buf,
    const bool * sent_models,
    const bool * stored_models,
    size_t in_feat, size_t n_expert, size_t world_size,
    CudaStreamManager* smgr) {
    // assert world_size > 1
    int recv_ptr = 0;
    /* TODO: may save for backward */
    long*expert_ptr = new long[n_expert * world_size];
    expert_ptr[0] = 0;
    for (size_t i = 1; i < n_expert * world_size; ++i) {
        expert_ptr[i] = expert_ptr[i - 1] + local_expert_count[i - 1];
    }

    for (size_t i = 0; i < n_expert; ++i) {
        NCCL_SAFE_CALL(ncclGroupStart());
        for (size_t j = 0; j < world_size; ++j) {
            int idx = i + j * n_expert;
            if (local_expert_count[idx]) {
                // model fetched from other node
                if (stored_models[idx]) {
                    checkCudaErrors(cudaMemcpyAsync(
                        input_buf + recv_ptr * in_feat,
                        local_input_buf + expert_ptr[idx] * in_feat,
                        local_expert_count[idx] * in_feat * sizeof(scalar_t),
                        cudaMemcpyDeviceToDevice,
                        smgr->stream(1)));
                    recv_ptr += local_expert_count[idx];
                } else {
                    NCCL_SAFE_CALL(ncclSend(
                        local_input_buf + expert_ptr[idx] * in_feat, 
                        local_expert_count[idx] * in_feat * sizeof(scalar_t),
                        ncclChar, 
                        j,
                        smgr->ncclcomm,
                        smgr->stream(0)));
                }
            }
            if (global_expert_count[idx] && !sent_models[idx]) {
                NCCL_SAFE_CALL(ncclRecv(
                        input_buf + recv_ptr * in_feat,
                        global_expert_count[idx] * in_feat * sizeof(scalar_t),
                        ncclChar,
                        j,
                        smgr->ncclcomm,
                        smgr->stream(0)));
                recv_ptr += global_expert_count[idx];
            }
        }
        NCCL_SAFE_CALL(ncclGroupEnd());
    }
    delete [] expert_ptr;
    smgr->sync(2);
}

template<typename scalar_t>
void fmoe_cuda_global_gather_impl(
    const scalar_t* output_buf,
    const long* local_expert_count,
    const long* global_expert_count,
    scalar_t* local_output_buf,
    const bool * sent_models,
    const bool * stored_models,
    size_t out_feat, size_t n_expert, size_t world_size,
    CudaStreamManager* smgr) {
    long send_ptr = 0;
    /* TODO: may save for backward */
    long *expert_ptr = new long[n_expert * world_size];
    expert_ptr[0] = 0;
    for (size_t i = 1; i < n_expert * world_size; ++i) {
        expert_ptr[i] = expert_ptr[i - 1] + local_expert_count[i - 1];
    }

    for (size_t i = 0; i < n_expert; ++i) {
        NCCL_SAFE_CALL(ncclGroupStart());
        for (size_t j = 0; j < world_size; ++j) {
            int idx = i + j * n_expert;
            if (global_expert_count[idx] && !sent_models[idx]) {
                NCCL_SAFE_CALL(ncclSend(
                    output_buf + send_ptr * out_feat,
                    global_expert_count[idx] * out_feat * sizeof(scalar_t),
                    ncclChar,
                    j,
                    smgr->ncclcomm,
                    smgr->stream(0)));
                send_ptr += global_expert_count[idx];
            }
            if (local_expert_count[idx]) {
                if (stored_models[idx]) {
                    checkCudaErrors(cudaMemcpyAsync(
                        local_output_buf + expert_ptr[idx] * out_feat,
                        output_buf + send_ptr * out_feat,
                        local_expert_count[idx] * out_feat * sizeof(scalar_t),
                        cudaMemcpyDeviceToDevice,
                        smgr->stream(1)));
                    send_ptr += local_expert_count[idx];
                } else{
                    NCCL_SAFE_CALL(ncclRecv(
                        local_output_buf + expert_ptr[idx] * out_feat, 
                        local_expert_count[idx] * out_feat * sizeof(scalar_t),
                        ncclChar, 
                        j,
                        smgr->ncclcomm,
                        smgr->stream(0)));
                }
            }
        }
        NCCL_SAFE_CALL(ncclGroupEnd());
    }
    delete [] expert_ptr;
    smgr->sync(2);
}

void fmoe_cuda_exchange_cache_info_impl(
    bool * sent_models,
    bool * stored_models,
    bool * will_broadcast,
    bool * broadcast,
    long num_expert,
    long world_size,
    CudaStreamManager * smgr) {
    
    int rank;
    NCCL_SAFE_CALL(ncclCommUserRank(smgr->ncclcomm, &rank));

    NCCL_SAFE_CALL(ncclGroupStart());
    // Sent caching information
    for (int i = 0; i < world_size; i++) {
        if (i == rank) {
            checkCudaErrors(cudaMemsetAsync(
                sent_models + num_expert * i,
                0,
                num_expert,
                smgr->stream(1)));
            continue;
        }
        
        NCCL_SAFE_CALL(ncclSend(
            sent_models + num_expert * i,
            num_expert,
            ncclChar,
            i,
            smgr->ncclcomm,
            smgr->stream(0)));
        
        NCCL_SAFE_CALL(ncclRecv(
            stored_models + num_expert * i,
            num_expert,
            ncclChar,
            i,
            smgr->ncclcomm,
            smgr->stream(0)));
    }
    NCCL_SAFE_CALL(ncclGroupEnd());

    // exchange broadcast information
    NCCL_SAFE_CALL(ncclAllGather(
        will_broadcast,
        broadcast,
        num_expert,
        ncclChar,
        smgr->ncclcomm,
        smgr->stream(0)));

    smgr->sync(2);
}

template<typename scalar_t>
void fmoe_cuda_model_exchange_impl(
    bool * sent_models,
    bool * stored_models,
    bool * broadcast,
    std::vector<torch::Tensor> local_params,
    std::vector<std::vector<torch::Tensor>> params,
    long num_expert,
    long world_size,
    CudaStreamManager * smgr) {
    
    int rank;
    NCCL_SAFE_CALL(ncclCommUserRank(smgr->ncclcomm, &rank));
    size_t amount_sent = 0, amount_received = 0;

    for (size_t i = 0; i < num_expert; i++) {

        NCCL_SAFE_CALL(ncclGroupStart());
        
        // size in bytes
        auto size = local_params[i].numel() * sizeof(scalar_t);
        
        for (size_t j = 0; j < world_size; j++) {
            if (j == rank) continue;

            size_t idx = i + j * num_expert;

            if (sent_models[idx] && !broadcast[i + rank * num_expert]) {
                scalar_t * param = local_params[i].data_ptr<scalar_t>();
                // printf("Send %ld %ld %ld\n", i,j,param_idx);
                NCCL_SAFE_CALL(ncclSend(
                    param,
                    size,
                    ncclChar,
                    j,
                    smgr->ncclcomm,
                    smgr->stream(0)));
                amount_sent += size;
            }

            if (stored_models[idx] && !broadcast[idx]) {
                scalar_t * param = params[j][i].data_ptr<scalar_t>();
                // printf("Recv %ld %ld %ld\n", i,j,param_idx);
                NCCL_SAFE_CALL(ncclRecv(
                    param,
                    size,
                    ncclChar,
                    j,
                    smgr->ncclcomm,
                    smgr->stream(0)));
                amount_received += size;
            }
        }
        
        NCCL_SAFE_CALL(ncclGroupEnd());
    }

    // send broadcasted data
    for (size_t i = 0; i < num_expert; i++) {
        for (size_t j = 0; j < world_size; j++) {
            auto idx = i + j * num_expert;
            if (broadcast[idx]) {
                scalar_t * param = j == rank ? local_params[i].data_ptr<scalar_t>() : params[j][i].data_ptr<scalar_t>();
                // std::cout << "broadcast idx " << idx << std::endl;
                NCCL_SAFE_CALL(ncclBroadcast(
                    param,
                    param,
                    local_params[i].numel() * sizeof(scalar_t),
                    ncclChar,
                    j,
                    smgr->ncclcomm,
                    smgr->stream(0)));
            }
        }
    }

    smgr->sync(1);
    // printf("Model exchange: Sent %ld Received %ld\n", amount_sent, amount_received);
}


template<typename scalar_t>
void fmoe_cuda_gradient_exchange_impl(
    bool * sent_models, 
    bool * stored_models, 
    long * expert_counts,
    std::vector<torch::Tensor> local_grads, 
    std::vector<std::vector<torch::Tensor>> grads, 
    long num_expert, 
    long world_size,
    CudaStreamManager * smgr) {

    int rank;
    NCCL_SAFE_CALL(ncclCommUserRank(smgr->ncclcomm, &rank));
    size_t amount_sent = 0, amount_received = 0;

    auto storage = std::vector<torch::Tensor>();

    // Creates the tensors with the required size according to how many models to fetch
    // ex: (2,3) sent to 4 nodes, creates (4,2,3)
    for (int i = 0; i < num_expert; i++) {
        auto sizes = local_grads[i].sizes();
        std::vector<long> shape;
        shape.push_back(expert_counts[i]);
        for (auto v : sizes) {
            shape.push_back(v);
        }
        
        c10::IntArrayRef x(shape);
        storage.push_back(local_grads[i].new_zeros(x));
    }

    for (int i = 0; i < num_expert; i++) {
        NCCL_SAFE_CALL(ncclGroupStart());
        
        // size in bytes
        auto size = local_grads[i].numel() * sizeof(scalar_t);
        
        int recv_ptr = 0;
        for (size_t j = 0; j < world_size; j++) {
            if (j == rank) continue;

            size_t idx = i + j * num_expert;
            auto count = local_grads[i].numel();

            if (sent_models[idx]) {
                scalar_t * param = storage[i].data_ptr<scalar_t>();
                NCCL_SAFE_CALL(ncclRecv(
                    param + recv_ptr * count,
                    size,
                    ncclChar,
                    j,
                    smgr->ncclcomm,
                    smgr->stream(0)));
                
                recv_ptr++;
                amount_received += size;
            }

            if (stored_models[idx]) {
                scalar_t * param = grads[j][i].data_ptr<scalar_t>();
                NCCL_SAFE_CALL(ncclSend(
                    param,
                    size,
                    ncclChar,
                    j,
                    smgr->ncclcomm,
                    smgr->stream(0)));
                amount_sent += size;
            }
        }
        
        NCCL_SAFE_CALL(ncclGroupEnd());
    }

    smgr->sync(1);
    checkCudaErrors(cudaGetLastError());
    // printf("Gradient exchange: Sent %ld Received %ld\n", amount_sent, amount_received);

    for (int i = 0; i < num_expert; i++) {
        storage[i] = storage[i].sum(0);
        local_grads[i].add_(storage[i]).div_(expert_counts[i] + 1); // TODO do we average them or just sum them?
    }
}

#endif  // FMOE_USE_NCCL
