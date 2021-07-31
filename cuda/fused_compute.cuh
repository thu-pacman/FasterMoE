#ifndef FUSED_COMPUTE_H
#define FUSED_COMPUTE_H

#include <cstdio>
#include <iostream>
#include <vector>


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
        long num_expert, long world_size,
        bool has_bias,
        CudaStreamManager* smgr) {

    int ptr = 0;
    int send_ptr = 0;
    int recv_ptr = 0;

    int *expert_ptr = new int[num_expert * world_size];
    expert_ptr[0] = 0;
    for (int i = 1; i < num_expert * world_size; ++i) {
        expert_ptr[i] = expert_ptr[i - 1] + local_expert_count[i - 1];
    }

    for (int ei = 0; ei < num_expert; ++ei) {
        int expert_count = 0;
        NCCL_SAFE_CALL(ncclGroupStart());
        for (int j = 0; j < world_size; ++j) {
            int idx = ei + j * num_expert;
            if (local_expert_count[idx]) {
                NCCL_SAFE_CALL(ncclSend(
                        input_buf + expert_ptr[idx] * d_model, 
                        local_expert_count[idx] * d_model * sizeof(scalar_t),
                        ncclChar, 
                        j,
                        smgr->ncclcomm,
                        smgr->stream(0)));
            }
            if (global_expert_count[idx]) {
                NCCL_SAFE_CALL(ncclRecv(
                        global_input_buf + recv_ptr * d_model,
                        global_expert_count[idx] * d_model * sizeof(scalar_t),
                        ncclChar,
                        j,
                        smgr->ncclcomm,
                        smgr->stream(0)));
                recv_ptr += global_expert_count[idx];
                expert_count += global_expert_count[idx];
            }
        }
        NCCL_SAFE_CALL(ncclGroupEnd());

        _compute_mlp_forward(
                global_input_buf,
                weight1,
                weight2,
                middle_buf,
                global_output_buf,
                has_bias,
                ei,
                ptr,
                expert_count,
                d_model,
                d_hidden,
                smgr->stream(0),
                smgr->handle(0));

        ptr += expert_count;

        NCCL_SAFE_CALL(ncclGroupStart());
        for (int j = 0; j < world_size; ++j) {
            int idx = ei + j * num_expert;
            if (global_expert_count[idx]) {
                NCCL_SAFE_CALL(ncclSend(
                        global_output_buf + send_ptr * d_model,
                        global_expert_count[idx] * d_model * sizeof(scalar_t),
                        ncclChar,
                        j,
                        smgr->ncclcomm,
                        smgr->stream(0)));
                send_ptr += global_expert_count[idx];
            }
            if (local_expert_count[idx]) {
                NCCL_SAFE_CALL(ncclRecv(
                        output_buf + expert_ptr[idx] * d_model, 
                        local_expert_count[idx] * d_model * sizeof(scalar_t),
                        ncclChar,
                        j,
                        smgr->ncclcomm,
                        smgr->stream(0)));
            }
        }
        NCCL_SAFE_CALL(ncclGroupEnd());
    }
    delete [] expert_ptr;
    smgr->sync(0);
}

#endif  // FUSED_COMPUTE_H
