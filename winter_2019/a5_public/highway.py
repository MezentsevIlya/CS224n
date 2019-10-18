#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
"""

### YOUR CODE HERE for part 1h
import torch
import torch.nn as nn
import torch.nn.functional as F


class Highway(nn.Module):
    """
    Highway model that maps from x_conv_out to x_highway
    """

    def __init__(self, embed_size):
        """
        Init layers
        :param embed_size: size of word embedding, equals to e_word
        """
        super(Highway, self).__init__()

        self.W_proj = nn.Linear(in_features=embed_size, out_features=embed_size, bias=True)
        self.W_gate = nn.Linear(in_features=embed_size, out_features=embed_size, bias=True)

    def forward(self, x_conv_out: torch.Tensor):
        """
        Mapping from x_conv_out to x_highway
        :param x_conv_out:  size: (batch_size, e_word), e_word=embed_size
        :return: x_highway, size: (batch_size, e_word), e_word=embed_size
        """
        x_proj = self.W_proj(x_conv_out)
        x_proj = F.relu(x_proj)

        x_gate = self.W_gate(x_conv_out)
        x_gate = torch.sigmoid(x_gate)

        x_highway = x_gate * x_proj + (1 - x_gate) * x_proj  # pointwise

        return x_highway


### END YOUR CODE


def test_highway():
    batch_size = 10
    print('batch_size: ', batch_size)
    e_word = 32

    print('x_conv_out must be size of (batch_size, e_word) = ', (batch_size, e_word))
    print('x_proj must be size of (batch_size, e_word) = ', (batch_size, e_word))
    print('x_gate must be size of (batch_size, e_word) = ', (batch_size, e_word))
    print('x_highway must be size of (batch_size, e_word) = ', (batch_size, e_word))

    x_conv_out = torch.randn(batch_size, e_word)

    hw = Highway(e_word)
    _ = hw.forward(x_conv_out)


if __name__ == '__main__':
    test_highway()
