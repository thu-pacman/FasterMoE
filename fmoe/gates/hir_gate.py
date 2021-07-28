from .naive_gate import NaiveGate

import torch
import torch.nn.functional as F


class HirGate(NaiveGate):
    def __init__(self, d_model, n_expert, world_size, node_rank):
        super().__init__(d_model, n_expert, world_size, top_k=2)
        self.ne_per_node = 8 * n_expert
        self.ogn_ratio = .22
        self.node_rank = node_rank

        mask = [0] * world_size * n_expert
        for i in range(n_expert * world_size):
            if i // self.ne_per_node != self.node_rank:
                mask[i] = 1
        self.mask = torch.Tensor(mask).bool()

    def forward(self, inp):
        if self.mask.device != inp.device:
            self.mask = self.mask.to(inp.device)

        gate_score = self.gate(inp)
        with torch.no_grad():
            top1_val, top1_idx = torch.topk(gate_score, k=1, dim=-1)
            # mask for outgoing tokens
            ogn_mask = (top1_idx // self.ne_per_node) != self.node_rank
            ogn_thres = int(inp.shape[0] * self.ogn_ratio)

        if ogn_mask.sum().item() < ogn_thres:
            topk_val, topk_idx = torch.topk(gate_score, k=self.top_k)
            topk_val = F.softmax(topk_val, dim=-1)
            return topk_idx, topk_val

        with torch.no_grad():
            _, top_ogn = torch.topk(top1_val.view(-1), k=ogn_thres)
            cand = gate_score.clone()
            cand[:, self.mask] = float('-inf')
            _, topk_idx = torch.topk(cand, k=self.top_k)
            topk_idx[top_ogn, 1] = top1_idx.view(-1)[top_ogn]

        idx_x = torch.arange(inp.shape[0], device=inp.device).repeat_interleave(2)
        topk_val = gate_score[idx_x, topk_idx.view(-1)].view(-1, self.top_k)

        topk_val = F.softmax(topk_val, dim=-1)
        return topk_idx, topk_val


def gen_hir_gate(rank):
    def _gen(d_model, n_expert, world_size, top_k=2):
        assert top_k == 2
        return HirGate(d_model, n_expert, world_size, rank)
    return _gen
