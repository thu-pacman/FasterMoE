#include "stream_manager.h"
#ifdef FMOE_USE_NCCL
#include <vector>
#include <torch/extension.h>


// given an all expert counts array with shape [world_size, world_size, num_experts], populates the equivalente global_expert_count as if done via all-to-all
__global__
void generate_global_counts(const long * all_expert_count, long * all_global_count, int n_expert, int world_size) {
    int worker_idx = blockIdx.x; // what worker am I
    int tid = threadIdx.x; // what position am i handling

    int offset = tid / n_expert; // what line should be written
    int pos = tid % n_expert; // the position in the line
    
    if (tid < n_expert * world_size) {
        all_global_count[worker_idx * world_size * n_expert + tid] = all_expert_count[offset * world_size * n_expert + worker_idx * n_expert + pos];
    }
}

void fmoe_cuda_expert_exchange_impl(
        const long* local_expert_count, 
        long* global_expert_count, 
        long * all_expert_count,
        long * all_global_expert_count,
        int n_expert, int world_size,
        CudaStreamManager* smgr) {
    
    int rank;
    NCCL_SAFE_CALL(ncclCommUserRank(smgr->ncclcomm, &rank));

    NCCL_SAFE_CALL(ncclAllGather(
        local_expert_count,
        all_expert_count,
        world_size * n_expert * sizeof(long),
        ncclChar,
        smgr->ncclcomm,
        smgr->stream(0)));

    dim3 grid_dim(world_size);
    dim3 block_dim(1024);
    generate_global_counts<<<grid_dim, block_dim, 0, smgr->stream(0)>>>(
        all_expert_count,
        all_global_expert_count,
        n_expert, 
        world_size
    );

    checkCudaErrors(cudaMemcpyAsync(
        global_expert_count,
        all_global_expert_count + rank * world_size * n_expert,
        world_size * n_expert * sizeof(long),
        cudaMemcpyDeviceToDevice,
        smgr->stream(0)));

    smgr->sync(1);
    checkCudaErrors(cudaGetLastError());
}

template<typename scalar_t>
void fmoe_cuda_global_scatter_impl(
    const scalar_t* local_input_buf,
    const long* local_expert_count,
    const long* global_expert_count,
    scalar_t* input_buf,
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
    
    int rank;
    NCCL_SAFE_CALL(ncclCommUserRank(smgr->ncclcomm, &rank));

    // Locally move the data to its correct place before any data is received/sent mantaining the tokens contiguous
    for (size_t j = 0; j < rank; j++) {
        for (size_t i = 0; i < n_expert; i++) {
            int idx = i + j * n_expert;
            if (stored_models[idx]) {
                checkCudaErrors(cudaMemcpyAsync(
                    input_buf + recv_ptr * in_feat,
                    local_input_buf + expert_ptr[idx] * in_feat,
                    local_expert_count[idx] * in_feat * sizeof(scalar_t),
                    cudaMemcpyDeviceToDevice,
                    smgr->stream(1)));
                recv_ptr += local_expert_count[idx];
            }
        }
    }

    // Receive data from other nodes
    for (size_t i = 0; i < n_expert; ++i) {
        NCCL_SAFE_CALL(ncclGroupStart());
        for (size_t j = 0; j < world_size; ++j) {
            int idx = i + j * n_expert;
            if (local_expert_count[idx] && (!stored_models[idx] || j == rank)) {
                // model fetched from other node
                NCCL_SAFE_CALL(ncclSend(
                    local_input_buf + expert_ptr[idx] * in_feat, 
                    local_expert_count[idx] * in_feat * sizeof(scalar_t),
                    ncclChar, 
                    j,
                    smgr->ncclcomm,
                    smgr->stream(0)));
            }
            if (global_expert_count[idx] && (!stored_models[rank * n_expert + i] || j == rank)) {
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

    // Handle the data of experts after this node
    for (size_t j = rank + 1; j < world_size; j++) {
        for (size_t i = 0; i < n_expert; i++) {
            int idx = i + j * n_expert;
            if (stored_models[idx]) {
                checkCudaErrors(cudaMemcpyAsync(
                    input_buf + recv_ptr * in_feat,
                    local_input_buf + expert_ptr[idx] * in_feat,
                    local_expert_count[idx] * in_feat * sizeof(scalar_t),
                    cudaMemcpyDeviceToDevice,
                    smgr->stream(1)));
                recv_ptr += local_expert_count[idx];
            }
        }
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

    int rank;
    NCCL_SAFE_CALL(ncclCommUserRank(smgr->ncclcomm, &rank));
    
    // Locally move the data to its correct place before any data is received/sent mantaining the tokens contiguous
    for (size_t j = 0; j < rank; j++) {
        for (size_t i = 0; i < n_expert; i++) {
            int idx = i + j * n_expert;
            if (stored_models[idx]) {
                checkCudaErrors(cudaMemcpyAsync(
                    local_output_buf + expert_ptr[idx] * out_feat,
                    output_buf + send_ptr * out_feat,
                    local_expert_count[idx] * out_feat * sizeof(scalar_t),
                    cudaMemcpyDeviceToDevice,
                    smgr->stream(1)));
                send_ptr += local_expert_count[idx];
            } 
        }
    }


    for (size_t i = 0; i < n_expert; ++i) {
        NCCL_SAFE_CALL(ncclGroupStart());
        for (size_t j = 0; j < world_size; ++j) {
            int idx = i + j * n_expert;
            if (local_expert_count[idx] && (!stored_models[idx] || j == rank)) {
                NCCL_SAFE_CALL(ncclRecv(
                    local_output_buf + expert_ptr[idx] * out_feat, 
                    local_expert_count[idx] * out_feat * sizeof(scalar_t),
                    ncclChar, 
                    j,
                    smgr->ncclcomm,
                    smgr->stream(0)));
            }

            if (global_expert_count[idx] && (!stored_models[rank * n_expert + i] || j == rank)) {
                NCCL_SAFE_CALL(ncclSend(
                    output_buf + send_ptr * out_feat,
                    global_expert_count[idx] * out_feat * sizeof(scalar_t),
                    ncclChar,
                    j,
                    smgr->ncclcomm,
                    smgr->stream(0)));
                send_ptr += global_expert_count[idx];
            }
        }
        NCCL_SAFE_CALL(ncclGroupEnd());
    }

    // Handle the data of experts after this node
    for (size_t j = rank + 1; j < world_size; j++) {
        for (size_t i = 0; i < n_expert; i++) {
            int idx = i + j * n_expert;
            if (stored_models[idx]) {
                checkCudaErrors(cudaMemcpyAsync(
                    local_output_buf + expert_ptr[idx] * out_feat,
                    output_buf + send_ptr * out_feat,
                    local_expert_count[idx] * out_feat * sizeof(scalar_t),
                    cudaMemcpyDeviceToDevice,
                    smgr->stream(1)));
                send_ptr += local_expert_count[idx];
            }
        }
    }

    delete [] expert_ptr;
    smgr->sync(2);
}

void fmoe_cuda_exchange_cache_info_impl(
    bool * sent_models,
    bool * stored_models,
    long num_expert,
    long world_size,
    CudaStreamManager * smgr) {

    // exchange broadcast information
    NCCL_SAFE_CALL(ncclAllGather(
        sent_models,
        stored_models,
        num_expert,
        ncclChar,
        smgr->ncclcomm,
        smgr->stream(0)));

    smgr->sync(1);
}

template<typename scalar_t>
void fmoe_cuda_model_exchange_impl(
    bool * stored_models,
    std::vector<torch::Tensor> local_params,
    std::vector<std::vector<torch::Tensor>> params,
    long num_expert,
    long world_size,
    CudaStreamManager * smgr) {
    int rank;
    NCCL_SAFE_CALL(ncclCommUserRank(smgr->ncclcomm, &rank));

    // send broadcasted data
    for (size_t i = 0; i < num_expert; i++) {
        for (size_t j = 0; j < world_size; j++) {
            auto idx = i + j * num_expert;
            if (stored_models[idx]) {
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
}


void exchange_reduce(float * param, int size, int j, CudaStreamManager * smgr) {
    NCCL_SAFE_CALL(ncclReduce(
        param,
        param,
        size,
        ncclFloat,
        ncclSum,
        j,
        smgr->ncclcomm,
        smgr->stream(0)));
}

void exchange_reduce(double * param, int size, int j, CudaStreamManager * smgr) {
    NCCL_SAFE_CALL(ncclReduce(
        param,
        param,
        size,
        ncclDouble,
        ncclSum,
        j,
        smgr->ncclcomm,
        smgr->stream(0)));
}

void exchange_reduce(c10::Half * param, int size, int j, CudaStreamManager * smgr) {
    NCCL_SAFE_CALL(ncclReduce(
        param,
        param,
        size,
        ncclHalf,
        ncclSum,
        j,
        smgr->ncclcomm,
        smgr->stream(0)));
}

template<typename scalar_t>
void fmoe_cuda_gradient_exchange_impl(
    bool * stored_models,
    std::vector<torch::Tensor> local_grads, 
    std::vector<std::vector<torch::Tensor>> grads, 
    long num_expert, 
    long world_size,
    CudaStreamManager * smgr) {

    int rank;
    NCCL_SAFE_CALL(ncclCommUserRank(smgr->ncclcomm, &rank));

    // send back broadcasted data
    for (size_t i = 0; i < num_expert; i++) {
        for (size_t j = 0; j < world_size; j++) {
            auto idx = i + j * num_expert;
            if (stored_models[idx]) {
                scalar_t * param = j == rank ? local_grads[i].data_ptr<scalar_t>() : grads[j][i].data_ptr<scalar_t>();

                auto size = local_grads[i].numel();
                exchange_reduce(param, size, j, smgr);
            }
        }
    }

    smgr->sync(1);
    checkCudaErrors(cudaGetLastError());
}

#endif  // FMOE_USE_NCCL
