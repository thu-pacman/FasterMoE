from .naive_gate import NaiveGate

import sys
import torch
import torch.distributed as dist
import torch.nn.functional as F
from .utils import limit_by_capacity
import fmoe_cuda
from fmoe.functions import count_by_gate


nw_per_node = 4

cr_cnt = 0
llims = [3, 2, 5, 2, 8, 2]


class HirGate(NaiveGate):
    def __init__(self, d_model, n_expert, world_size, node_rank):
        global cr_cnt
        self.rep_lim = llims[cr_cnt % len(llims)]# 4 - cr_cnt % 4
        cr_cnt += 1
        super().__init__(d_model, n_expert, world_size, top_k=2)
        self.ne_per_node = nw_per_node * n_expert
        self.ogn_ratio = .14
        self.node_rank = node_rank

        mask = [0] * world_size * n_expert
        for i in range(n_expert * world_size):
            if i // self.ne_per_node == self.node_rank:
                mask[i] = 1
        self.mask = torch.Tensor(mask).bool()
        self.stored_models = None
        self.policy_fn = None

    def forward(self, inp):
        if self.mask.device != inp.device:
            self.mask = self.mask.to(inp.device)

        gate_score = self.gate(inp)
        lim_mask = self.mask

        # if self.stored_models is not None:
            # lim_mask = lim_mask | self.stored_models.view(-1).to(lim_mask.device)
        lim_mask = ~lim_mask

        top2_val, top2_idx = torch.topk(gate_score, k=2, dim=-1)
        S = gate_score.shape[0]
        top_k = 2

        with torch.no_grad():
            top1_idx = top2_idx.view((-1, top_k))[:, 0]
            top1_val = top2_val.view((-1, top_k))[:, 0]
        c_e = torch.scatter_add(
                torch.zeros(self.tot_expert, device=top1_idx.device),
                0,
                top1_idx,
                torch.ones_like(top1_idx, dtype=torch.float),
                ) / S
        m_e = torch.mean(F.softmax(gate_score, dim=1), dim=0)
        loss = torch.mean(c_e * m_e) * (self.num_expert ** 2)
        self.set_loss(loss)

        with torch.no_grad():
            _, lec, aec, gec, agec = count_by_gate(top2_idx, 
                    self.num_expert, self.world_size, require_pos=False)
            stored_models = self.policy_fn(aec, agec,
                    self.num_expert, self.world_size, inp.shape[-1], True)
            # if stored_models.sum().item() < self.rep_lim:
            lim_mask = lim_mask & ~stored_models.view(-1).to(lim_mask.device)

            # mask for outgoing tokens
            ogn_mask = lim_mask[top1_idx]
            ogn_thres = int(inp.shape[0] * self.ogn_ratio)

        if ogn_mask.sum().item() < ogn_thres:
            topk_val, topk_idx = torch.topk(gate_score, k=self.top_k)
            topk_val = F.softmax(topk_val, dim=-1)
            return topk_idx, topk_val

        with torch.no_grad():
            # sys.stderr.write('stored {}\n'.format(self.stored_models))
            # sys.stderr.write('lim_mask {}\n'.format(lim_mask))
            # sys.stderr.write('ogn mask {}\n'.format(ogn_mask))
            # sys.stderr.write('top1 val {}\n'.format(top1_val))
            top1_val[~ogn_mask] = float('-inf')
            _, top_ogn = torch.topk(top1_val.view(-1), k=ogn_thres)
            cand = gate_score.clone()
            cand[:, lim_mask] = float('-inf')
            _, topk_idx = torch.topk(cand, k=self.top_k)
            # sys.stderr.write(f'{inp.shape}\n')
            # sys.stderr.write(f'{top1_idx.shape}\n')
            # sys.stderr.write(f'{ogn_mask.shape}\n')
            # sys.stderr.write(f'{top_ogn.max()} {top_ogn.shape}\n')
            # sys.stderr.write(f'{topk_idx}\n')
            topk_idx[top_ogn, 1] = top1_idx.view(-1)[top_ogn]

        idx_x = torch.arange(inp.shape[0], device=inp.device).repeat_interleave(2)
        topk_val = gate_score[idx_x, topk_idx.view(-1)].view(-1, self.top_k)

        # sys.stderr.write('{}: exceeding limits by {} / {}\n'.format(
        #     dist.get_rank(), ogn_mask.sum().item(), ogn_thres))
        # local_expert_count = torch.zeros(
        #     self.num_expert * self.world_size, device=topk_val.device, dtype=torch.int32
        # )
        # fmoe_cuda.expert_count(topk_idx, local_expert_count)
        # local_expert_count = local_expert_count.long().cpu()
        # sys.stderr.write('{}: lec {}\n'.format(dist.get_rank(), local_expert_count))

        # capacity = int(1.2 * inp.shape[0] * self.top_k)
        # _new_lec, _new_gec, topk_idx = limit_by_capacity(
        #         topk_idx, self.num_expert, self.world_size, capacity)

        topk_val = F.softmax(topk_val, dim=-1)

        return topk_idx, topk_val


def gen_hir_gate(rank):
    def _gen(d_model, n_expert, world_size, top_k=2):
        assert top_k == 2
        return HirGate(d_model, n_expert, world_size, rank // nw_per_node)
    return _gen
