r"""
Adaption to act as the MLP layer using an MoE MLP layer in transformer.
"""
import torch
import torch.nn as nn
from .gates import NaiveGate
from .layers import FMoE, FMoELinear


class _Expert(nn.Module):
    r"""
    An expert using 2 FMoELinear modules to speed up the computation of experts
    within one worker.
    """

    def __init__(self, num_expert, d_model, d_hidden, activation, rank=0, bias=True):
        super().__init__()
        self.htoh4 = FMoELinear(num_expert, d_model, d_hidden, bias=bias, rank=rank)
        self.h4toh = FMoELinear(num_expert, d_hidden, d_model, bias=bias, rank=rank)
        self.activation = activation

    def forward(self, inp, fwd_expert_count):
        r"""
        First expand input to 4h (the hidden size is variable, but is called h4
        for convenience). Then perform activation. Finally shirink back to h.
        """
        x = self.htoh4(inp, fwd_expert_count)
        x = self.activation(x)
        x = self.h4toh(x, fwd_expert_count)
        return x


def _gen_policy(alpha):
    def _global_policy(all_experts_count, all_global_expert_count, num_expert, world_size, d_model, fused):
        bw_pcie = 88 * 1e9 / 8
        bw_net = 50 * 1e9 / 8
        bw_mm = 11.5e12
        data_size = 4 # TODO data different than float

        if fused:
            all_experts_count = all_experts_count.sum(dim=-1).view(world_size, world_size, 1)
            all_global_expert_count = all_global_expert_count.sum(dim=-1).view(world_size, world_size, 1)

        fwd_expert_counts = all_global_expert_count.sum(1) # [world_size, num_expert]
        default_counts = fwd_expert_counts.clone()

        _, indices = fwd_expert_counts.sort(0, descending=True)

        alphaHsquared = alpha * d_model ** 2 * data_size

        B_w = default_counts.max(0)[0]
        lat_comp = 3 * 4 * B_w * alphaHsquared / bw_mm  + 4 * B_w * d_model * data_size / bw_net

        comm = float('+inf')
        model_size = 2 * alphaHsquared * num_expert / bw_net * 2
        comp_time = 12 * alphaHsquared / bw_mm

        for i, index in enumerate(indices):
            fwd_expert_counts[index] = 0
            fwd_expert_counts += all_global_expert_count[index].view(world_size, -1)

            B_k = fwd_expert_counts.max(0)[0]
            lat_comm = fwd_expert_counts.max(0)[0] * comp_time + (i+1) * model_size

            if lat_comm < comm:
                comm = lat_comm
            elif lat_comm > comm:
                break

        res = all_experts_count.new_zeros(world_size, num_expert, dtype=bool)

        if lat_comp > comm:
            res[indices[:i]] = True
        return res

    def _no_policy(all_experts_count, all_global_expert_count, num_expert, world_size, d_model, fused):
        if fused:
            all_experts_count = all_experts_count.sum(dim=-1).view(world_size, world_size, 1)
        res = all_experts_count.new_zeros(world_size, num_expert, dtype=bool)
        return res

    import os
    if os.environ.get('FMOE_ENABLE_DYNREP', '0') == '1':
        return _global_policy
    else:
        return _no_policy


class FMoETransformerMLP(FMoE):
    r"""
    A complete MoE MLP module in a Transformer block.
    * `activation` is the activation function to be used in MLP in each expert.
    * `d_hidden` is the dimension of the MLP layer.
    """

    def __init__(
        self,
        num_expert=32,
        d_model=1024,
        d_hidden=4096,
        world_size=1,
        mp_group=None,
        moe_group=None,
        activation=torch.nn.GELU(),
        gate=NaiveGate,
        top_k=2,
        expert_dp_comm="none",
        gate_hook=None,
        bias=True,
        mask=None,
        mask_dict=None,
    ):
        super().__init__(
            num_expert=num_expert,
            d_model=d_model,
            gate=gate,
            top_k=top_k,
            world_size=world_size,
            mp_group=mp_group,
            moe_group=moe_group,
            gate_hook=gate_hook,
            mask=mask,
            policy_fn=_gen_policy(d_hidden / d_model),
            mask_dict=mask_dict
        )
        self.experts = _Expert(
            num_expert, d_model, d_hidden, activation, rank=self.mp_rank, bias=bias
        )
        self.mark_parallel_comm(expert_dp_comm)

    def forward(self, inp: torch.Tensor):
        r"""
        This module wraps up the FMoE module with reshape, residual and layer
        normalization.
        """
        original_shape = inp.shape
        inp = inp.reshape(-1, self.d_model)
        output = super().forward(inp)
        return output.reshape(original_shape)
