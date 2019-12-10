# _*_coding:utf-8_*_
# user  : dudaizhong
# time  : 19-12-10 下午8:22
# info  :
from numpy.ma import copy
from torch import nn


def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])
