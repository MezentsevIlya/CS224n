#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

### YOUR CODE HERE for part 1i
class CNN(nn.Module):
    def __init__(self, e_char, filters_num, kernel_size):
        super(CNN, self).__init__()

        self.conv = nn.Conv1d(in_channels=e_char, out_channels=filters_num, kernel_size=kernel_size,
                              stride=1, padding=0)
        print('e_char: ', e_char)
        print('filters_num: ', filters_num)
        print('kernel_size: ', 5)


    def forward(self, x_reshaped):
        print('x_reshaped: ', x_reshaped.size())

        x_conv = self.conv(x_reshaped)

        print('x_conv: ', x_conv.size())

        x_conv_out = F.relu(x_conv)
        x_conv_out = F.max_pool1d(x_conv_out, kernel_size=x_conv.size()[-1])
        x_conv_out = torch.squeeze(x_conv_out, dim=2)
        print('x_conv_out: ', x_conv_out.size())

        return x_conv_out
### END YOUR CODE


def test_cnn():
    batch_size = 10
    print('batch_size: ', batch_size)
    e_char = 50
    m_word = 20

    filter_num = 32
    kernel_size = 5

    print('x_reshaped must be size of (batch_size, e_char, m_word) = ', (batch_size, e_char, m_word))
    print('x_conv must be size of (batch_size, f, m_word - k + 1) = ', (batch_size, filter_num, m_word - kernel_size + 1))
    print('x_conv_out must be size of (batch_size, e_word (=f)) = ', (batch_size, filter_num))

    x_reshaped = torch.randn(batch_size, e_char, m_word)

    cnn = CNN(e_char, filter_num, kernel_size)
    _ = cnn.forward(x_reshaped)


if __name__ == '__main__':
    test_cnn()

