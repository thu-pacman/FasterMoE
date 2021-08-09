#ifndef FUSED_COMPUTE_H
#define FUSED_COMPUTE_H

#include <cstdio>
#include <iostream>
#include <vector>
#include <thread>


#include <cuda.h>
#include <cuda_runtime.h>
#include <nccl.h>

#include "utils/cublas_wrapper.h"
#include "utils/fmoe_utils.h"
#include "stream_manager.h"

#define SMGR_N_STREAMS 16

template<typename scalar_t>
__global__ 
void relu_kernel(scalar_t* a, size_t n) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    scalar_t v;
    for (; i < n; i += stride) {
        v = a[i];
        if (v < 0) {
            a[i] = 0;
        }
    }
}

template<typename scalar_t>
__global__ 
void relu_backward_kernel(const scalar_t* a, scalar_t* grad_o, size_t n) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    for (; i < n; i += stride) {
        if (a[i] <= 0) {
            grad_o[i] = 0;
        }
    }
}

#define NTH 512

template<typename scalar_t>
void _compute_mlp_forward(
        const scalar_t* input_buf,
        const scalar_t* weight1,
        const scalar_t* weight2,
        scalar_t* middle_buf,
        scalar_t* output_buf,
        const bool has_bias,
        const size_t expert_idx,
        const size_t batch_offset,
        const size_t batch_size,
        const size_t d_model,
        const size_t d_hidden,
        cudaStream_t stream,
        cublasHandle_t handle) {
    scalar_t alpha = 1, beta = has_bias ? 1 : 0; 
    if (batch_size == 0) {
        return;
    }
    size_t in_feat, out_feat;
    in_feat = d_model, out_feat = d_hidden;
    checkCudaErrors(cublasXgemm(
            handle,
            CUBLAS_OP_T,
            CUBLAS_OP_N,
            out_feat, batch_size, in_feat,
            &alpha,
            weight1 + expert_idx * in_feat * out_feat, in_feat,
            input_buf + batch_offset * in_feat, in_feat,
            &beta,
            middle_buf + batch_offset * out_feat, out_feat
            ));

    relu_kernel<<<CEIL(batch_size * d_hidden, NTH), NTH, 0, stream>>>
        (middle_buf + batch_offset * d_hidden, batch_size * d_hidden);

    in_feat = d_hidden, out_feat = d_model;
    checkCudaErrors(cublasXgemm(
            handle,
            CUBLAS_OP_T,
            CUBLAS_OP_N,
            out_feat, batch_size, in_feat,
            &alpha,
            weight2 + expert_idx * in_feat * out_feat, in_feat,
            middle_buf + batch_offset * in_feat, in_feat,
            &beta,
            output_buf + batch_offset * out_feat, out_feat
            ));
}


template<typename scalar_t>
void _compute_mlp_backward(
        const scalar_t* input_buf,
        const scalar_t* weight1,
        const scalar_t* weight2,
        const scalar_t* middle_buf,
        const scalar_t* output_buf,
        const scalar_t* grad_out,

        scalar_t* grad_middle,
        scalar_t* grad_weight1,
        scalar_t* grad_weight2,
        scalar_t* grad_in,

        const bool has_bias,
        const size_t expert_idx,
        const size_t batch_offset,
        const size_t batch_size,
        const size_t d_model,
        const size_t d_hidden,

        bool is_first,
        cudaStream_t stream,
        cublasHandle_t handle) {
    scalar_t alpha = 1, beta_weight = is_first ? 0 : 1, beta_feat = 0; 
    if (batch_size == 0) {
        return;
    }
    size_t in_feat, out_feat;
    in_feat = d_hidden, out_feat = d_model;
    // Backward input: g_m = w2 @ g_o
    checkCudaErrors(cublasXgemm(
            handle,
            CUBLAS_OP_N,
            CUBLAS_OP_N,
            in_feat, batch_size, out_feat,
            &alpha,
            weight2 + expert_idx * in_feat * out_feat, in_feat,
            grad_out + batch_offset * out_feat, out_feat,
            &beta_feat,
            grad_middle + batch_offset * in_feat , in_feat
            ));

    // Backward weight: g_w2 = m @ g_o
    checkCudaErrors(cublasXgemm(
            handle,
            CUBLAS_OP_N,
            CUBLAS_OP_T,
            in_feat, out_feat, batch_size,
            &alpha,
            middle_buf + batch_offset * in_feat, in_feat,
            grad_out + batch_offset * out_feat, out_feat,
            &beta_weight,
            grad_weight2 + expert_idx * in_feat * out_feat, in_feat
            ));

    relu_backward_kernel<<<CEIL(batch_size * d_hidden, NTH), NTH, 0, stream>>>
        (middle_buf + batch_offset * d_hidden,
         grad_middle + batch_offset * d_hidden,
         batch_size * d_hidden);

    in_feat = d_model, out_feat = d_hidden;
    // Backward input: g_i = w1 @ g_m
    checkCudaErrors(cublasXgemm(
            handle,
            CUBLAS_OP_N,
            CUBLAS_OP_N,
            in_feat, batch_size, out_feat,
            &alpha,
            weight1 + expert_idx * in_feat * out_feat, in_feat,
            grad_middle + batch_offset * out_feat, out_feat,
            &beta_feat,
            grad_in + batch_offset * in_feat , in_feat
            ));

    // Backward weight: g_w1 = i @ g_m
    checkCudaErrors(cublasXgemm(
            handle,
            CUBLAS_OP_N,
            CUBLAS_OP_T,
            in_feat, out_feat, batch_size,
            &alpha,
            input_buf + batch_offset * in_feat, in_feat,
            grad_middle + batch_offset * out_feat, out_feat,
            &beta_weight,
            grad_weight1 + expert_idx * in_feat * out_feat, in_feat
            ));
}


template<typename scalar_t>
void _exchange_with(
        const scalar_t* sendbuf, size_t sendcount, int t_send,
        scalar_t* recvbuf, size_t recvcount, int t_recv,
        long d_model,
        cudaStream_t stream, ncclComm_t comm) {
    if (sendcount) {
        ncclSend(sendbuf, sendcount * d_model * sizeof(scalar_t),
                ncclChar, t_send , comm, stream);
    }
    if (recvcount) {
        ncclRecv(recvbuf, recvcount * d_model * sizeof(scalar_t),
                ncclChar, t_recv, comm, stream);
    }
}


#define GEN_BASE(_step) \
    long to_base = (group_rank + _step) % n_groups * pipeline_gran; \
    long from_base = (group_rank + n_groups - _step) % n_groups * pipeline_gran;
#define GEN_IDX \
    int idx_send = ei + rank_send * num_expert; \
    int idx_recv = ei + rank_recv * num_expert; \
    int gidx_send = ei * world_size + rank_send; \
    int gidx_recv = ei * world_size + rank_recv; \
    int idx_self = ei +      rank * num_expert;

void _compute_ptrs(long num_expert, long rank, long world_size, 
        const long* local_expert_count, 
        const long* global_expert_count, 
        const bool* stored_models,
        int *local_ptr,
        int *global_ptr,
        int *local_global_ptr) {
    local_ptr[0] = global_ptr[0] = local_global_ptr[0] = 0;
    
    for (int i = 0; i < num_expert * world_size; ++i) {
        local_ptr[i + 1] = local_ptr[i] + local_expert_count[i];

        local_global_ptr[i + 1] = local_global_ptr[i];
        // if model fetched, add local tokens
        if (stored_models[i]){
            local_global_ptr[i + 1] += local_expert_count[i];
        }

        auto expert_idx = i % num_expert;
        auto worker_idx = i / num_expert;
        auto gp_idx = expert_idx * world_size + worker_idx;
        // if local model wasn't fetched, receive global tokens
        if (stored_models[rank * num_expert + expert_idx]) {
            global_ptr[gp_idx + 1] = 0;
        } else {
            global_ptr[gp_idx + 1] = global_expert_count[i];
        }
    }
    global_ptr[0] = 0;
    for (int i = 0; i < num_expert * world_size; ++i) {
        global_ptr[i + 1] += global_ptr[i];
    }
}

template<typename scalar_t>
void fmoe_cuda_fused_forward_impl(
        const scalar_t* input_buf,
        std::vector<std::vector<std::vector<torch::Tensor>>> params,

        scalar_t* global_input_buf,
        scalar_t* middle_buf,
        scalar_t* global_output_buf,
        scalar_t* output_buf,

        const long* local_expert_count, 
        const long* global_expert_count, 
        const bool* stored_models,

        long d_model, long d_hidden, 
        long num_expert, long rank, long world_size,
        bool has_bias,
        long pipeline_gran, CudaStreamManager* smgr) {

    int *local_ptr = new int[num_expert * world_size + 1];
    int *global_ptr = new int[num_expert * world_size + 1];
    int *local_global_ptr = new int[num_expert * world_size + 1]; // local fetched models tracker
    _compute_ptrs(num_expert, rank, world_size,
            local_expert_count, global_expert_count, stored_models,
            local_ptr, global_ptr, local_global_ptr);

    if (pipeline_gran > world_size) {
        pipeline_gran = world_size;
    }
    long n_groups = world_size / pipeline_gran;
    long group_rank = rank / pipeline_gran;

    cudaEvent_t *input_ready = new cudaEvent_t[n_groups];
    cudaEvent_t *output_ready = new cudaEvent_t[n_groups];
    for (long i = 0; i < n_groups; ++i) {
        cudaEventCreate(input_ready + i);
        cudaEventCreate(output_ready + i);
    }

    for (long step = 0; step < n_groups; ++step) {
        for (int ei = 0; ei < num_expert; ++ei) {
            GEN_BASE(step);
            NCCL_SAFE_CALL(ncclGroupStart());
            for (int j = 0; j < pipeline_gran; ++j) {
                int rank_send = j + to_base;
                int rank_recv = j + from_base;
                GEN_IDX;
                _exchange_with(input_buf + local_ptr[idx_send] * d_model,
                        local_expert_count[idx_send] * !stored_models[idx_send], rank_send,
                        global_input_buf + global_ptr[gidx_recv] * d_model,
                        global_expert_count[idx_recv] * !stored_models[idx_self], rank_recv,
                        d_model, smgr->stream(0), smgr->ncclcomm);
            }
            NCCL_SAFE_CALL(ncclGroupEnd());
        }
        cudaEventRecord(input_ready[step], smgr->stream(0));
    }

    int last = params[rank][0].size() / 2; // bias = False
    scalar_t * weight1 = params[rank][0][0].data_ptr<scalar_t>();
    scalar_t * weight2 = params[rank][0][last].data_ptr<scalar_t>();

    for (long step = 0; step < n_groups; ++step) {
        cudaStreamWaitEvent(smgr->stream(1), input_ready[step], 0);
        for (int ei = 0; ei < num_expert; ++ei) {
            GEN_BASE(step);
            long offset = global_ptr[ei * world_size + from_base];
            long micro_batch_size = global_ptr[ei * world_size + 
                (from_base + pipeline_gran)] - offset;
            
            _compute_mlp_forward(
                    global_input_buf, weight1, weight2,
                    middle_buf, global_output_buf,
                    has_bias,
                    ei,
                    offset, micro_batch_size,
                    d_model, d_hidden,
                    smgr->stream(1), smgr->handle(1));
        }
        cudaEventRecord(output_ready[step], smgr->stream(1));
    }

    for (long step = 0; step < n_groups; ++step) {
        cudaStreamWaitEvent(smgr->stream(0), output_ready[step], 0);
        for (int ei = 0; ei < num_expert; ++ei) {
            GEN_BASE(step);
            NCCL_SAFE_CALL(ncclGroupStart());
            for (int j = 0; j < pipeline_gran; ++j) {
                int rank_send = j + from_base;
                int rank_recv = j + to_base;
                GEN_IDX;
                _exchange_with(global_output_buf + global_ptr[gidx_send] * d_model,
                        global_expert_count[idx_send] * !stored_models[idx_self], rank_send,
                        output_buf + local_ptr[idx_recv] * d_model,
                        local_expert_count[idx_recv] * !stored_models[idx_recv], rank_recv,
                        d_model, smgr->stream(0), smgr->ncclcomm);
            }
            NCCL_SAFE_CALL(ncclGroupEnd());
        }
    }

    // TODO local movement + computation
    int offset = global_ptr[world_size * num_expert];
    for (int j = 0; j < world_size; j++) {
        
        for (int i = 0; i < num_expert; i++) {
            int idx = j * num_expert + i;
            if (!stored_models[idx])
                continue;
            weight1 = params[j][0][0].data_ptr<scalar_t>();
            weight2 = params[j][0][last].data_ptr<scalar_t>();

            auto stream = 2 + (idx % (SMGR_N_STREAMS- 2));

            _compute_mlp_forward(
                input_buf + local_ptr[idx] * d_model, weight1, weight2,
                middle_buf + (offset + local_global_ptr[idx]) * d_hidden, output_buf + local_ptr[idx] * d_model,
                has_bias,
                i,
                0, local_expert_count[idx],
                d_model, d_hidden,
                smgr->stream(stream), smgr->handle(stream));

        }
    }


    delete [] local_ptr;
    delete [] global_ptr;
    delete [] local_global_ptr;
    smgr->sync(SMGR_N_STREAMS+1);
    checkCudaErrors(cudaGetLastError());
    for (long i = 0; i < n_groups; ++i) {
        cudaEventDestroy(input_ready[i]);
        cudaEventDestroy(output_ready[i]);
    }
    delete [] input_ready;
    delete [] output_ready;
}


template<typename scalar_t>
void fmoe_cuda_fused_backward_impl(
        const scalar_t* input_buf,
        const scalar_t* original_input_buf,
        std::vector<std::vector<std::vector<torch::Tensor>>> params,
        const scalar_t* middle_buf,
        const scalar_t* output_buf,
        const scalar_t* grad_out,

        scalar_t* global_grad_out,
        scalar_t* global_grad_in,

        scalar_t* grad_middle,
        scalar_t* grad_in,

        const long* local_expert_count, 
        const long* global_expert_count, 
        const bool* stored_models,
        long d_model, long d_hidden, 
        long num_expert, long rank, long world_size,
        bool has_bias,
        long pipeline_gran, CudaStreamManager* smgr) {

    int *local_ptr = new int[num_expert * world_size + 1];
    int *global_ptr = new int[num_expert * world_size + 1];
    int *local_global_ptr = new int[num_expert * world_size + 1]; // local fetched models tracker

    _compute_ptrs(num_expert, rank, world_size,
            local_expert_count, global_expert_count, stored_models,
            local_ptr, global_ptr, local_global_ptr);
   
    if (pipeline_gran > world_size) {
        pipeline_gran = world_size;
    }
    long n_groups = world_size / pipeline_gran;
    long group_rank = rank / pipeline_gran;

    cudaEvent_t *input_ready = new cudaEvent_t[n_groups];
    cudaEvent_t *output_ready = new cudaEvent_t[n_groups];
    for (long i = 0; i < n_groups; ++i) {
        cudaEventCreate(input_ready + i);
        cudaEventCreate(output_ready + i);
    }

    for (long step = 0; step < n_groups; ++step) {
        for (int ei = 0; ei < num_expert; ++ei) {
            GEN_BASE(step);
            NCCL_SAFE_CALL(ncclGroupStart());
            for (int j = 0; j < pipeline_gran; ++j) {
                int rank_send = j + to_base;
                int rank_recv = j + from_base;
                GEN_IDX;
                _exchange_with(grad_out + local_ptr[idx_send] * d_model,
                        local_expert_count[idx_send] * !stored_models[idx_send], rank_send,
                        global_grad_out + global_ptr[gidx_recv] * d_model,
                        global_expert_count[idx_recv] * !stored_models[idx_self], rank_recv,
                        d_model, smgr->stream(0), smgr->ncclcomm);
            }
            NCCL_SAFE_CALL(ncclGroupEnd());
        }
        cudaEventRecord(input_ready[step], smgr->stream(0));
    }

    int last = params[rank][0].size() / 2; // bias = False
    scalar_t * weight1 = params[rank][0][0].data_ptr<scalar_t>();
    scalar_t * weight2 = params[rank][0][last].data_ptr<scalar_t>();
    scalar_t * grad_weight1 = params[rank][0][0].mutable_grad().data_ptr<scalar_t>();
    scalar_t * grad_weight2 = params[rank][0][last].mutable_grad().data_ptr<scalar_t>();
    
    for (long step = 0; step < n_groups; ++step) {
        cudaStreamWaitEvent(smgr->stream(1), input_ready[step], 0);
        for (int ei = 0; ei < num_expert; ++ei) {
            GEN_BASE(step);
            long offset = global_ptr[ei * world_size + from_base];
            long micro_batch_size = global_ptr[ei * world_size + 
                (from_base + pipeline_gran)] - offset;

            _compute_mlp_backward(
                    input_buf, weight1, weight2,
                    middle_buf, output_buf, global_grad_out,
                    grad_middle, grad_weight1, grad_weight2, global_grad_in,
                    has_bias,
                    ei,
                    offset, micro_batch_size,
                    d_model, d_hidden, step == 0,
                    smgr->stream(1), smgr->handle(1));
        }
        cudaEventRecord(output_ready[step], smgr->stream(1));
    }

    for (long step = 0; step < n_groups; ++step) {
        cudaStreamWaitEvent(smgr->stream(0), output_ready[step], 0);
        for (int ei = 0; ei < num_expert; ++ei) {
            GEN_BASE(step);
            NCCL_SAFE_CALL(ncclGroupStart());
            for (int j = 0; j < pipeline_gran; ++j) {
                int rank_send = j + from_base;
                int rank_recv = j + to_base;
                GEN_IDX;
                _exchange_with(global_grad_in + global_ptr[gidx_send] * d_model,
                        global_expert_count[idx_send] * !stored_models[idx_self], rank_send,
                        grad_in + local_ptr[idx_recv] * d_model,
                        local_expert_count[idx_recv] * !stored_models[idx_recv], rank_recv,
                        d_model, smgr->stream(0), smgr->ncclcomm);
            }
            NCCL_SAFE_CALL(ncclGroupEnd());
        }
    }

    checkCudaErrors(cudaGetLastError());

    int offset = global_ptr[world_size * num_expert];
    for (int j = 0; j < world_size; j++) {
        
        for (int i = 0; i < num_expert; i++) {
            int idx = j * num_expert + i;
            if (!stored_models[idx])
                continue;
            
            weight1 = params[j][0][0].data_ptr<scalar_t>();
            weight2 = params[j][0][last].data_ptr<scalar_t>();    
            grad_weight1 = params[j][0][0].mutable_grad().data_ptr<scalar_t>();
            grad_weight2 = params[j][0][last].mutable_grad().data_ptr<scalar_t>();
            
            auto stream = 2 + (idx % (SMGR_N_STREAMS- 2));

            _compute_mlp_backward(
                original_input_buf + local_ptr[idx] * d_model, weight1, weight2,
                middle_buf + (offset + local_global_ptr[idx]) * d_hidden, output_buf, grad_out + local_ptr[idx] * d_model,
                grad_middle + (offset + local_global_ptr[idx]) * d_hidden, grad_weight1, grad_weight2, grad_in + local_ptr[idx] * d_model,
                has_bias,
                i,
                0, local_expert_count[idx],
                d_model, d_hidden, 0, // we never consider it to be the first since it's already initialized to zero and we are lazy
                smgr->stream(stream), smgr->handle(stream));

        }
    }


    delete [] local_ptr;
    delete [] global_ptr;
    delete [] local_global_ptr;
    smgr->sync(SMGR_N_STREAMS+1);
    checkCudaErrors(cudaGetLastError());
    for (long i = 0; i < n_groups; ++i) {
        cudaEventDestroy(input_ready[i]);
        cudaEventDestroy(output_ready[i]);
    }
    delete [] input_ready;
    delete [] output_ready;
}

#endif  // FUSED_COMPUTE_H
