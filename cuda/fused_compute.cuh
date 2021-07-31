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
void relu_backward_kernel(scalar_t* a, const scalar_t* grad_o, 
        scalar_t* grad_i, size_t n) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    scalar_t v;
    for (; i < n; i += stride) {
        if (a[i] <= 0) {
            v = 0;
        } else {
            v = grad_o[i];
        }
        grad_i[i] = v;
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
    int idx_recv = ei + rank_recv * num_expert;

template<typename scalar_t>
void fmoe_cuda_fused_forward_impl(
        const scalar_t* input_buf,
        const scalar_t* weight1,
        const scalar_t* weight2,

        scalar_t* global_input_buf,
        scalar_t* middle_buf,
        scalar_t* global_output_buf,
        scalar_t* output_buf,

        const long* local_expert_count, 
        const long* global_expert_count, 
        long d_model, long d_hidden, 
        long num_expert, long rank, long world_size,
        bool has_bias,
        long pipeline_gran, CudaStreamManager* smgr) {

    int *local_ptr = new int[num_expert * world_size + 1];
    int *global_ptr = new int[num_expert * world_size + 1];
    local_ptr[0] = global_ptr[0] = 0;
    for (int i = 1; i <= num_expert * world_size; ++i) {
        local_ptr[i] = local_ptr[i - 1] + local_expert_count[i - 1];
        global_ptr[i] = global_ptr[i - 1] + global_expert_count[i - 1];
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
                        local_expert_count[idx_send], rank_send,
                        global_input_buf + global_ptr[idx_recv] * d_model,
                        global_expert_count[idx_recv], rank_recv,
                        d_model, smgr->stream(0), smgr->ncclcomm);
            }
            NCCL_SAFE_CALL(ncclGroupEnd());
        }
        cudaEventRecord(input_ready[step], smgr->stream(0));
    }

    long compute_step = 0;

    std::thread th([&]() {
        for (long step = 0; step < n_groups; compute_step = ++step) {
            cudaEventSynchronize(input_ready[step]);
            for (int ei = 0; ei < num_expert; ++ei) {
                GEN_BASE(step);
                long offset = global_ptr[from_base * num_expert];
                long micro_batch_size = global_ptr[(from_base + pipeline_gran) * num_expert] - offset;
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
    });

    for (long step = 0; step < n_groups; ++step) {
        while (step >= compute_step);
        for (int ei = 0; ei < num_expert; ++ei) {
            GEN_BASE(step);
            NCCL_SAFE_CALL(ncclGroupStart());
            for (int j = 0; j < pipeline_gran; ++j) {
                int rank_send = j + from_base;
                int rank_recv = j + to_base;
                GEN_IDX;
                _exchange_with(global_output_buf + global_ptr[idx_send] * d_model,
                        global_expert_count[idx_send], rank_send,
                        output_buf + local_ptr[idx_recv] * d_model,
                        local_expert_count[idx_recv], rank_recv,
                        d_model, smgr->stream(0), smgr->ncclcomm);
            }
            NCCL_SAFE_CALL(ncclGroupEnd());
        }
    }

    delete [] local_ptr;
    delete [] global_ptr;
    delete [] input_ready;
    delete [] output_ready;
    th.join();
    smgr->sync(0);
}

#endif  // FUSED_COMPUTE_H
