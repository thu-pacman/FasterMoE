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
            inp, models,
            pos_s, pos_g,
            local_expert_count, global_expert_count,
            stored_models,
            fwd_batch_size, out_batch_size,
            world_size):
        local_input_buf = _local_scatter(inp, pos_s)

        model_params = [[tuple(m.parameters()) for m in node] for node in models]
        local_output_buf, gib, gmb, gob = fmoe_cuda.fused_forward(
                local_input_buf, model_params,
                local_expert_count, global_expert_count, 
                stored_models, fwd_batch_size,
                world_size, False)

        out = _local_gather(local_output_buf, pos_g, out_batch_size,
                maybe_overlap=False)
        
        variables = (pos_s, pos_g, local_expert_count, global_expert_count,
                gib, gmb, gob, stored_models, local_input_buf)
        
        ctx.moe_args = fwd_batch_size, inp.shape[0], world_size, model_params
        ctx.save_for_backward(*variables)

        return out

    @staticmethod
    def backward(ctx, grad_out):
        (pos_s, pos_g, local_expert_count, global_expert_count,
                gib, gmb, gob, stored_models, local_input_buf) = ctx.saved_tensors
        (fwd_batch_size, inp_batch_size, world_size, model_params) = ctx.moe_args

        grad_out_buf = _local_scatter(grad_out.contiguous(), pos_g)
        (grad_in_buf, )  = fmoe_cuda.fused_backward(
                gib, model_params, gmb, gob, grad_out_buf,
                local_expert_count, global_expert_count,
                local_input_buf,
                stored_models,
                fwd_batch_size, pos_s.shape[0],
                world_size, False)
        # print('We reached here so everything should be fine')
        # torch.distributed.barrier()
        grad_in = _local_gather(grad_in_buf, pos_s, inp_batch_size)

        return (grad_in, None, None, None, None, None, None, 
                None, None, None)
