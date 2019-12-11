# _*_coding:utf-8_*_
# user  : dudaizhong
# time  : 19-12-11 下午3:35
# info  :
from torch import nn

from model.transformer.LayerNorm import LayerNorm
from model.transformer.utils import clones


class Decoder(nn.Module):

    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)

        return self.norm(x)
