# _*_coding:utf-8_*_
# user  : dudaizhong
# time  : 19-12-11 下午3:36
# info  :
from torch import nn

from model.transformer.SublayerConnection import SublayerConnection
from model.transformer.utils import clones


class DecoderLayer(nn.Module):
    """ Self-attention + encoder self-attention + feed forward """

    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()

        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x, memory, src_mask, tgt_mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, memory, memory, src_mask))

        return self.sublayer[2](x, self.feed_forward)
