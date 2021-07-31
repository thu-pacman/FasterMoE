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
                inp, weight1, weight2,
                local_expert_count, global_expert_count, fwd_batch_size,
                world_size, False)
        out = _local_gather(local_output_buf, pos_g, out_batch_size,
                maybe_overlap=False)

        variables = (pos_s, pos_g, local_expert_count, global_expert_count,
                inp, weight1, weight2, gib, gmb, gob)
        ctx.save_for_backward(*variables)

        return out

    @staticmethod
    def backward(ctx, grad_out):
        raise NotImplementedError
