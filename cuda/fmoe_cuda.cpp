#include <iostream>
#include <vector>
#include <torch/extension.h>

// global_exchange
#ifdef FMOE_USE_NCCL
#include <c10d/ProcessGroupNCCL.hpp>
std::vector<torch::Tensor> _expert_exchange(
        torch::Tensor local_expert_count,
        long n_expert, long n_workers);
torch::Tensor _global_scatter(
        torch::Tensor input_buf,
        torch::Tensor local_expert_count,
        torch::Tensor global_expert_count,
        torch::Tensor stored_models,
        long batch_size, long n_workers);
torch::Tensor _global_gather(
        torch::Tensor output_buf,
        torch::Tensor local_expert_count,
        torch::Tensor global_expert_count,
        torch::Tensor stored_models,
        long batch_size, long n_workers);
//TODO remove this function
std::vector<torch::Tensor> _exchange_cache_info(
        torch::Tensor sent_models,
        long num_expert,
        long world_size);
int _model_exchange(
        torch::Tensor stored_models,
        std::vector<torch::Tensor> local_params,
        std::vector<std::vector<torch::Tensor>> params,
        long num_expert, long world_size);
torch::Tensor _generate_cached_count(
        torch::Tensor local_expert_count,
        torch::Tensor global_expert_count,
        torch::Tensor stored_models,
        long num_expert, long world_size);
void _gradient_exchange(
        torch::Tensor stored_models,
        std::vector<torch::Tensor> local_grads,
        std::vector<std::vector<torch::Tensor>> grads,
        long num_expert, long world_size);

void _ensure_nccl(c10d::ProcessGroupNCCL& p, torch::Tensor t);
#endif  // FMOE_USE_NCCL

// local_exchange
void _assign_pos(
        torch::Tensor cum_count,
        torch::Tensor gate,
        torch::Tensor pos);
void _expert_count(
        torch::Tensor gate_idx,
        torch::Tensor expert_count);

// parallel_linear
torch::Tensor _linear_forward(
        torch::Tensor input_buf,
        torch::Tensor expert_count,
        torch::Tensor weight,
        at::optional<torch::Tensor> bias
        );
std::vector<torch::Tensor> _linear_backward(
        torch::Tensor grad_output_buf,
        torch::Tensor input_buf,
        torch::Tensor expert_count,
        torch::Tensor weight,
        at::optional<torch::Tensor> bias
        );

// balancing
torch::Tensor _limit_by_capacity(
        torch::Tensor expert_count, torch::Tensor capacity,
        long n_expert, long n_experts);
torch::Tensor _prune_gate_by_capacity(
        torch::Tensor gate_idx, torch::Tensor expert_count,
        long n_expert, long n_worker);

// fused functions
std::vector<torch::Tensor> _fused_forward(
        torch::Tensor input_buf,
        std::vector<std::vector<std::vector<torch::Tensor>>> params,
        torch::Tensor local_expert_count,
        torch::Tensor global_expert_count,
        torch::Tensor stored_models,
        long global_batch_size,
        long n_workers, bool has_bias);
std::vector<torch::Tensor> _fused_backward(
        torch::Tensor input_buf,
        std::vector<std::vector<std::vector<torch::Tensor>>> params,
        torch::Tensor middle_buf,
        torch::Tensor output_buf,
        torch::Tensor grad_out,
        torch::Tensor local_expert_count,
        torch::Tensor global_expert_count,
        torch::Tensor inp,
        torch::Tensor stored_models,
        long global_batch_size,
        long buf_batch_size,
        long n_workers, bool has_bias);

// base_layer
torch::Tensor _balanced_assignment(torch::Tensor job_and_worker_to_score);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
#ifdef FMOE_USE_NCCL
    m.def("expert_exchange", &_expert_exchange, "FastMoE expert exchange (CUDA)");
    m.def("global_scatter", &_global_scatter, "FastMoE global scatter (CUDA)");
    m.def("global_gather", &_global_gather, "FastMoE global gather (CUDA)");
    m.def("ensure_nccl", &_ensure_nccl, "FastMoE ensure torch nccl comm");

    m.def("fused_forward", &_fused_forward, "FastMoE fuse global exchange and linear forward");
    m.def("fused_backward", &_fused_backward, "FastMoE fuse global exchange and linear backward");
    m.def("exchange_cache_info", &_exchange_cache_info, "FastMoE exchange cache info (CUDA)");
    m.def("model_exchange", &_model_exchange, "FastMoE model exchange (CUDA)");
    m.def("generate_cached_count", &_generate_cached_count, "FastMoE generate cached count (CUDA)");
    m.def("gradient_exchange", &_gradient_exchange, "FastMoE gradient exchange (CUDA)");
#endif

    m.def("expert_count", &_expert_count, "FastMoE count gate indices (CUDA)");
    m.def("assign_pos", &_assign_pos, "FastMoE assign pos by gate (CUDA)");

    m.def("linear_forward", &_linear_forward, "FastMoE forward (CUDA)");
    m.def("linear_backward", &_linear_backward, "FastMoE backward (CUDA)");

    m.def("limit_by_capacity", &_limit_by_capacity, "FastMoE limit experts by capacity(CUDA)");
    m.def("prune_gate_by_capacity", &_prune_gate_by_capacity, "FastMoE prune gate by capacity(CUDA)");

    m.def("balanced_assignment", &_balanced_assignment, "Balanced Assignment");
}
