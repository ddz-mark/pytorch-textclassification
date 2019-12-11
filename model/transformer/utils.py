# _*_coding:utf-8_*_
# user  : dudaizhong
# time  : 19-12-10 下午8:22
# info  :
import copy
import torch

from torch import nn
import numpy as np


def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def subsequent_mask(size):
    """
    "Mask out subsequent positions."
    在训练期间，当前解码位置的词不能Attend到后续位置的词。
    :param size:
    :return:
    """
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0
