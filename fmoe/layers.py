r"""
Layers that FMoE provides to users
"""
import torch
import torch.nn as nn
import math
import os

from .functions import prepare_forward, ensure_comm
from .functions import MOEScatter, MOEGather, MOELinear, MOECache
from .functions import AllGather, Slice
from .fused_functions import MOEForward
from .gates import NaiveGate


class FMoELinear(nn.Module):
    r"""
    A linear layer that contains multiple experts.
    As multiple experts can be placed on the same worker, the computation can be
    performed in parallel to increase the performance.
    The FMoELinear module provides such function.
    """

    def __init__(
        self,
        num_expert: int,
        in_feat: int,
        out_feat: int,
        bias: bool = True,
        rank: int = 0,
    ):
        super().__init__()
        self.num_expert = num_expert
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.rank = rank
        self.weight = nn.Parameter(torch.Tensor(num_expert, out_feat, in_feat))
        if bias:
            self.bias = nn.Parameter(torch.zeros(num_expert, out_feat))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def forward(self, inp, fwd_expert_count):
        r"""
        Call MOE function
        """
        x = MOELinear.apply(inp, fwd_expert_count, self.weight, self.bias)
        return x

    def extra_repr(self) -> str:
        return "num_expert={}, in_features={}, \
        out_features={}, bias={}, rank={}".format(
            self.num_expert,
            self.in_feat,
            self.out_feat,
            self.bias is not None,
            self.rank,
        )

    def reset_parameters(self):
        # Approach is the same as in torch.nn.Linear
        # https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/linear.py#L88
        # bias is left to zero, similar as megatron

        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))


def mark_module_parallel_comm(module, comm):
    r"""
    Mark all parameters in `module` as doing data parallel in `comm`, where
    `comm` may be one of `'world', 'dp', 'none'`.
    """
    for p in module.parameters():
        setattr(p, "dp_comm", comm)


def _fmoe_general_global_forward(
    inp, 
    gate, 
    expert_fn, 
    policy_fn, 
    experts, 
    num_expert, 
    world_size, 
    fused, 
    enable_fuse=False):
    r"""
    A private function that performs the following steps to complete the MoE
    computation.
    * Count the number of tokens from each worker to each expert.
    * Send the features to their target position so that input features to each
    expert are contiguous in memory.
    * Perform the forward computation of the experts using `expert_fn`
    * Gather the output features of experts back, and reorder them as sentences.
    Intermediate results like expert counts are hidden from users by this
    function.
    """
    (
        pos,
        local_expert_count,
        global_expert_count,
        all_expert_count,
        all_global_expert_count,
        fwd_expert_count,
    ) = prepare_forward(gate, num_expert, world_size)

    topk = 1
    if len(gate.shape) == 2:
        topk = gate.shape[1]

    out_batch_size = inp.shape[0]
    if len(gate.shape) == 2:
        out_batch_size *= gate.shape[1]

    inp, models, stored_models, fwd_expert_count, fwd_batch_size = MOECache.apply(
        inp,
        all_expert_count.view(world_size, world_size, num_expert),
        all_global_expert_count.view(world_size, world_size, num_expert),
        local_expert_count.view(world_size, num_expert), 
        global_expert_count.view(world_size, num_expert), 
        fwd_expert_count, policy_fn, experts,
        inp.shape[0], inp.shape[1], topk, num_expert, world_size, fused)

    if enable_fuse:
        x = MOEForward.apply(
                inp, models,
                pos // topk, pos,
                local_expert_count, global_expert_count,
                stored_models,
                fwd_expert_count.sum().item(), out_batch_size, world_size)
        return x, stored_models

    x = MOEScatter.apply(
        inp, pos // topk,
        local_expert_count, global_expert_count, 
        stored_models,
        fwd_batch_size, world_size
    )
    x = expert_fn(x, models, fwd_expert_count, num_expert)

    x = MOEGather.apply(
        x, pos,
        local_expert_count, global_expert_count,
        stored_models,
        out_batch_size, world_size
    )
    return x, stored_models

class FMoE(nn.Module):
    r"""
    A general moe implementation that supports an arbitrary module as the
    expert.
    * `num_expert` stands for the number of experts on **each** worker.
    * `world_size` stands for the total number of workers that contains
    different experts.
    * `mp_group` can be a torch's communication group, indicating that model
    parallel is applied across the group, which means that workers in the group
    hold the same copy of the input feature, and demands the same copy of the
    output. FMoE saves computation by slicing the input in the mp group and
    performing all-gather after the MLP computation.
    * `top_k` stands for the number of experts each token is going to.
    * `gate` is a gate class which can found in `fmoe.gates`.
    * `expert` can be specified as a module class, it is used to generate
    `num_expert` expert modules.
    """

    def __init__(
        self,
        num_expert=32,
        d_model=1024,
        world_size=1,
        mp_group=None,
        moe_group=None,
        top_k=2,
        gate=NaiveGate,
        expert=None,
        gate_hook=None,
        policy_fn=None,
        mask=None,
        mask_dict=None,
        enable_fuse=False
    ):
        super().__init__()
        self.num_expert = num_expert
        self.d_model = d_model
        self.world_size = world_size
        self.mp_group = mp_group
        if mp_group is None:
            self.mp_size = 1
            self.mp_rank = 0
        else:
            self.mp_size = mp_group.size()
            self.mp_rank = mp_group.rank()
        self.top_k = top_k
        if type(expert) is list:
            self.experts = nn.ModuleList([e(d_model) for e in expert])
            self.experts_fused = False
            self.num_expert = num_expert = len(expert)
        elif expert is not None:
            self.experts = nn.ModuleList([expert(d_model)
                for _ in range(num_expert)])
            self.experts_fused = False
        else:
            self.experts_fused = True
        self.gate = gate(d_model, num_expert, world_size, top_k)
        if hasattr(self.gate, 'stored_models'):
            self.gate.policy_fn = policy_fn
        self.gate_hook = gate_hook
        self.mask = mask
        self.mask_dict = mask_dict
        self.moe_group = moe_group

        if os.environ.get('FMOE_ENABLE_FUSE', '0') == '1':
            enable_fuse = True
        self.enable_fuse = enable_fuse
        if enable_fuse:
            assert(self.experts_fused)
        self.policy_fn=policy_fn

    def expert_fn(self, inp, models, fwd_expert_count, num_expert):
        r"""
        The default expert function which either calls the experts as a whole
        or as separate experts.
        """
        res = []
        
        input_ptr = 0
        for j, counts in enumerate(fwd_expert_count.view(-1, num_expert)):
            size = counts.sum().item()

            node_inp = inp[input_ptr:input_ptr + size]
            input_ptr += size
            experts = models[j] # node j's experts

            if not experts:
                if counts.any():
                    raise ValueError(f'Experts is {experts} yet counts is {counts}')
                continue
                
            if self.experts_fused:
                res.append(experts[0](node_inp, counts))
                continue
            
            outputs = []
            base_idx = 0
            for i in range(self.num_expert):
                batch_size = counts[i].item()
                inp_slice = node_inp[base_idx : base_idx + batch_size]
                outputs.append(experts[i](inp_slice))
                base_idx += batch_size
            res.append(torch.cat(outputs, dim=0))
        
        return torch.cat(res, dim=0)

    def mark_parallel_comm(self, expert_dp_comm="none"):
        r"""
        Automatically mark the data parallel comms of the parameters within the
        module. This can be typically called at the end of the __init__ function
        in child classes.
        """
        if self.experts is not None:
            comm = expert_dp_comm
            if isinstance(self.experts, list):
                for e in self.experts:
                    mark_module_parallel_comm(e, comm)
            else:
                mark_module_parallel_comm(self.experts, comm)
        mark_module_parallel_comm(self.gate, "moe")

    def forward(self, inp):
        r"""
        The FMoE module first computes gate output, and then conduct MoE forward
        according to the gate.  The score of the selected gate given by the
        expert is multiplied to the experts' output tensors as a weight.
        """
        if self.world_size > 1:
            ensure_comm(inp, self.moe_group)
        if self.mp_size > 1:
            inp = Slice.apply(inp, self.mp_rank, self.mp_size, self.mp_group)

        gate_top_k_idx, gate_score = self.gate(inp)

        if self.gate_hook is not None:
            self.gate_hook(gate_top_k_idx, gate_score, None)

        # delete masked tensors
        if self.mask is not None and self.mask_dict is not None:
            mask = self.mask.view(-1)
            # to: (BxL') x d_model
            inp = inp[mask == 0, :]
            gate_top_k_idx = gate_top_k_idx[mask == 0, :]


        fwd, stored_models = _fmoe_general_global_forward(
            inp, gate_top_k_idx,
            self.expert_fn, 
            self.policy_fn, 
            self.experts, 
            self.num_expert, 
            self.world_size, 
            self.experts_fused,
            self.enable_fuse
        )


        # recover deleted tensors
        if self.mask is not None and self.mask_dict is not None:
            # to: (BxL') x top_k x d_model
            fwd = fwd.view(-1, self.top_k, self.d_model)
            # to: (BxL) x top_k x d_model
            x = torch.zeros(mask.shape[0], self.top_k, self.d_model, device=fwd.device, dtype=fwd.dtype)
            # recover
            x[mask == 0] = fwd
            for k, v in self.mask_dict.items():
                x[mask == k] = v
        else:
            x = fwd.view(-1, self.top_k, self.d_model)

        gate_score = gate_score.view(x.shape[0], 1, self.top_k)
        x = torch.bmm(gate_score, x).reshape(-1, self.d_model)

        if self.mp_size > 1:
            x = AllGather.apply(x, self.mp_rank, self.mp_size, self.mp_group)
        return x
