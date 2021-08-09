import torch
import torch.distributed as dist
import fmoe
from fmoe.transformer import FMoETransformerMLP

from test_numerical import _assert_numerical


def test_forward(n_experts, d_model, d_hidden, batch_size, world_size):
    model = FMoETransformerMLP(
            n_experts, d_model, d_hidden, world_size,
            activation=torch.nn.ReLU(), top_k=1, bias=False).cuda()
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

    os.environ['FMOE_ENABLE_DYNREP'] = '1' # enable cache
    model.policy_fn = fmoe.transformer._gen_policy(d_model / d_hidden)
    print('-' * 15)
    out = model(inp)
    out.sum().backward()
    
    moe_g_in = inp.grad
    moe_g_w1 = model.experts.htoh4.weight.grad.detach()
    moe_g_w2 = model.experts.h4toh.weight.grad.detach()
    
    _assert_numerical(['forward', 'grad in', 'grad w1', 'grad w2'], 
            [out, moe_g_in, moe_g_w1, moe_g_w2],
            [std, std_g_in, std_g_w1, std_g_w2], rank)


if __name__ == '__main__':
    import os
    os.environ['RANK'] = os.environ.get("OMPI_COMM_WORLD_RANK", "0")
    os.environ["WORLD_SIZE"] = os.environ.get("OMPI_COMM_WORLD_SIZE", "1")
    assert int(os.environ["WORLD_SIZE"]) > 1, "Cache requires a distributed environment"
    
    os.environ['FMOE_ENABLE_DYNREP'] = '0' # disable cache

    dist.init_process_group(backend='nccl')
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    n_experts = 2
    d_model = 4
    d_hidden = 8
    batch_size = 64

    test_forward(n_experts, d_model, d_hidden, batch_size, world_size)
