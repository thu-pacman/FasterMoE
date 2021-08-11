# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch.nn as nn
import torch
import fmoe_cuda
import numpy as np


class BaseLayerGate(nn.Module):

    def __init__(self, d_model, num_expert, world_size, base_shuffle=True):
        super().__init__()
        self.world_size = world_size
        self.num_workers = num_expert
        self.num_expert = num_expert
        expert_centroids = torch.empty(self.num_workers, d_model).cuda()
        torch.nn.init.orthogonal_(expert_centroids, gain=0.1)
        self.register_parameter("expert_centroids", torch.nn.Parameter(expert_centroids))
        self.shuffle = base_shuffle

    def forward(self, input_features):
        features = input_features.reshape(-1, input_features.size(-1))
        is_training = input_features.requires_grad

        if self.shuffle and is_training:
            # Send each token to a random worker, to break correlations within the batch
            shuffle_sort = torch.randperm(features.size(0), device=features.device)
            features = All2All.apply(features[shuffle_sort])

        with torch.no_grad():
            # Compute similarity of each token to each expert, for routing
            token_expert_affinities = features.matmul(self.expert_centroids.transpose(0, 1))

        # Compute which token goes to which expert
        top_idx = self.balanced_assignment(token_expert_affinities)
        top_idx = np.delete(top_idx.cpu(), -1, axis=1).cuda()

        # Swap these tokens for the right ones for our expert
        top_value = torch.empty(top_idx.size())
        for i in range(0, self.num_expert):
            for j in range(0, int(int(features.size(0)) / self.num_expert)):
                top_value[i][j] = token_expert_affinities[top_idx[i][j]][i]

        return top_idx, top_value

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
