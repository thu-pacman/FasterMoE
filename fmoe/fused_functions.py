r"""
The fmoe.fused_functions module contains fused forward function
"""

import torch
from torch.autograd import Function
import fmoe_cuda
from .utils import get_torch_default_comm
from .functions import _local_scatter, _local_gather


class MOEForward(Function):
    @staticmethod
    def forward(
            ctx,
            inp, weight1, weight2,
            pos_s, pos_g,
            local_expert_count, global_expert_count,
            fwd_batch_size, out_batch_size,
            world_size):
        local_input_buf = _local_scatter(inp, pos_s)
        local_output_buf, gib, gmb, gob = fmoe_cuda.fused_forward(
                local_input_buf, weight1, weight2,
                local_expert_count, global_expert_count, fwd_batch_size,
                world_size, False)
        out = _local_gather(local_output_buf, pos_g, out_batch_size,
                maybe_overlap=False)

        variables = (pos_s, pos_g, local_expert_count, global_expert_count,
                weight1, weight2, gib, gmb, gob)
        ctx.moe_args = fwd_batch_size, inp.shape[0], world_size
        ctx.save_for_backward(*variables)

        return out

    @staticmethod
    def backward(ctx, grad_out):
        (pos_s, pos_g, local_expert_count, global_expert_count,
                weight1, weight2, gib, gmb, gob) = ctx.saved_tensors
        (fwd_batch_size, inp_batch_size, world_size) = ctx.moe_args

        grad_out_buf = _local_scatter(grad_out.contiguous(), pos_g)
        grad_in_buf, grad_weight1, grad_weight2 = fmoe_cuda.fused_backward(
                gib, weight1, weight2, gmb, gob, grad_out_buf,
                local_expert_count, global_expert_count,
                fwd_batch_size, pos_s.shape[0],
                world_size, False)
        grad_in = _local_gather(grad_in_buf, pos_s, inp_batch_size)

        return (grad_in, grad_weight1, grad_weight2, None, None, None, None, 
                None, None, None)
