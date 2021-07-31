import torch
import torch.distributed as dist
import fmoe
from fmoe.transformer import FMoETransformerMLP

from test_numerical import _assert_numerical


def test_forward(n_experts, d_model, d_hidden, batch_size, world_size):
    model = FMoETransformerMLP(
            n_experts, d_model, d_hidden, world_size,
            activation=torch.nn.ReLU(), bias=False).cuda()
    inp = torch.rand(batch_size, d_model).cuda()
    inp.requires_grad = True
    model.train()
    std = model(inp)
    std.sum().backward()
    std_g_in = inp.grad.detach()
    inp.grad = None
    std_g_w1 = model.experts.htoh4.weight.grad.detach()
    model.experts.htoh4.weight.grad = None
    std_g_w2 = model.experts.h4toh.weight.grad.detach()
    model.experts.h4toh.weight.grad = None

    model.enable_fuse = True
    model.expert_fn = model.experts
    out = model(inp)
    out.sum().backward()
    moe_g_in = inp.grad
    moe_g_w1 = model.experts.htoh4.weight.grad.detach()
    moe_g_w2 = model.experts.h4toh.weight.grad.detach()
    _assert_numerical(['forward', 'grad in', 'grad w1', 'grad w2'], 
            [out, moe_g_in, moe_g_w1, moe_g_w2],
            [std, std_g_in, std_g_w1, std_g_w2], rank)


if __name__ == '__main__':
    dist.init_process_group(backend='nccl')
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    n_experts = 1
    d_model = 4
    d_hidden = 8
    batch_size = 6

    test_forward(n_experts, d_model, d_hidden, batch_size, world_size)
