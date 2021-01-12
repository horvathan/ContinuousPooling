import torch
import torch.nn as nn
import torch.nn.functional as func

import numpy as np


class ContinuousPool(nn.Module):
    def __init__(self, in_channels, batch_length, timesteps=10, type='avg', ksize=2, strides=2, padding=0):
        super().__init__()
        self.in_channels = in_channels
        self.batch_size = batch_length
        self.timesteps = timesteps
        self.pool_type = type
        self.ksize = ksize
        self.strides = strides
        self.padding = padding

        self.pool_strength = nn.Parameter(torch.Tensor(np.full((1, self.in_channels, 1, 1), 0.1)))

    def forward(self, input):
        current_shape = input.size()
        pool_strength = self.pool_strength.repeat((current_shape[0], 1, current_shape[2], current_shape[3]))

        new_input = input.float()

        for i in range(self.timesteps):
            pool_op = nn.MaxPool2d(kernel_size=3, stride=(1, 1), padding=(1, 1))
            pooled = pool_op(new_input)
            diff = pooled - new_input
            shapes = {'new_input': new_input.size(), 'pooled': pooled.size(), 'diff': diff.size(), 'pool_strength': pool_strength.size()}
            new_input = new_input + (pool_strength * diff)

        if self.pool_type == 'avg':
            result_pool_op = nn.AvgPool2d(kernel_size=self.ksize, stride=self.strides, padding=self.padding)
        elif self.pool_type == 'max':
            result_pool_op = nn.MaxPool2d(kernel_size=self.ksize, stride=self.strides, padding=self.padding)
        else:
            raise ValueError('The available pooling types are \'avg\' and \'max\'!')

        return result_pool_op(new_input)
