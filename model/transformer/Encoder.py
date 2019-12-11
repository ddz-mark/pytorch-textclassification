# _*_coding:utf-8_*_
# user  : dudaizhong
# time  : 19-12-10 下午8:21
# info  :
from torch import nn

from model.transformer.LayerNorm import LayerNorm
from model.transformer.utils import clones


class Encoder(nn.Module):
    "Core encoder is a stack of N layers"

    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask=None):
        "Pass the input (and mask) through each layer in turn."
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)
