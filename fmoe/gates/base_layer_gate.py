# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch.nn as nn
import torch
import fmoe_cuda
import numpy as np


class BaseLayerGate(nn.Module):

    def __init__(self, d_model, num_expert, world_size, topk=2, base_shuffle=True):
        super().__init__()
        self.topk = topk
        self.world_size = world_size
        self.num_workers = world_size * num_expert
        self.num_expert = world_size * num_expert
        self.expert_centroids = []
        for i in range(topk):
            c = torch.empty(self.num_workers, d_model).cuda()
            torch.nn.init.orthogonal_(c, gain=0.1)
            self.register_parameter("expert_centroids_{}".format(i),
                    torch.nn.Parameter(c))
            self.expert_centroids.append(c)
        self.shuffle = base_shuffle

    def forward(self, input_features):
        features = input_features.reshape(-1, input_features.size(-1))
        is_training = input_features.requires_grad

        # if self.shuffle and is_training:
        #     # Send each token to a random worker, to break correlations within the batch
        #     shuffle_sort = torch.randperm(features.size(0), device=features.device)
        #     features = All2All.apply(features[shuffle_sort])

        tis, tvs = [], []
        for i in range(self.topk):
            print(i)
            with torch.no_grad():
                # Compute similarity of each token to each expert, for routing
                token_expert_affinities = features.matmul(self.expert_centroids[i].transpose(0, 1))
            print(token_expert_affinities)

            # Compute which token goes to which expert
            top_idx = self.balanced_assignment(token_expert_affinities)
            top_idx = np.delete(top_idx.cpu(), -1, axis=1).cuda()

            print(top_idx)

            # Swap these tokens for the right ones for our expert
            top_value = torch.empty(top_idx.size())
            for i in range(0, self.num_expert):
                for j in range(0, int(int(features.size(0)) / self.num_expert)):
                    top_value[i][j] = token_expert_affinities[top_idx[i][j]][i]
            tis.append(top_idx)
            tvs.append(top_value)

        return tis, tvs

    def balanced_assignment(self, scores):
        ok = scores.isfinite()
        if not ok.all():
            # NaNs here can break the assignment algorithm
            scores[~ok] = scores[ok].min()
        return fmoe_cuda.balanced_assignment(scores)



# Wraps torch.distributed.all_to_all_single as a function that supports autograd
class All2All(torch.autograd.Function):
    @staticmethod
    def forward(ctx, xs, input_splits=None, output_splits=None):
        ctx.input_splits = input_splits
        ctx.output_splits = output_splits

        ys = torch.empty_like(xs) if output_splits is None else \
            xs.new_empty(size=[sum(output_splits)] + list(xs.size()[1:]))
        torch.distributed.all_to_all_single(ys, xs, output_split_sizes=output_splits, input_split_sizes=input_splits)
        return ys

    @staticmethod
    def backward(ctx, grad_output):
        result = torch.empty_like(grad_output) if ctx.input_splits is None else \
            grad_output.new_empty(size=[sum(ctx.input_splits)] + list(grad_output.size()[1:]))
        torch.distributed.all_to_all_single(result, grad_output,
                                            output_split_sizes=ctx.input_splits, input_split_sizes=ctx.output_splits)
        return result, None, None
