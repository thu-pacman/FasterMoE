import torch
import torch.distributed as dist
import fmoe
from fmoe.transformer import FMoETransformerMLP

from .test_numerical import _assert_numerical


def test_forward(n_experts, d_model, d_hidden, batch_size, world_size):
    model = FMoETransformerMLP(
            n_experts, d_model, d_hidden, world_size).cuda()
    inp = torch.rand(batch_size, d_model)
    std = model(inp)
    model.enable_fuse = True
    model.expert_fn = model.experts
    out = model(inp)
    _assert_numerical('forward result', [out], [std], rank)


if __name__ == '__main__':
    dist.init_process_group(backend='nccl')
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    n_experts = 1
    d_model = 4
    d_hidden = 8
    batch_size = 6

    test_forward(n_experts, d_model, d_hidden, batch_size, world_size)
