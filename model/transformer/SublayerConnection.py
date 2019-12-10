# _*_coding:utf-8_*_
# user  : dudaizhong
# time  : 19-12-10 下午8:24
# info  :
from torch import nn

from model.transformer.LayerNorm import LayerNorm


class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        """
        "Apply residual connection to any sublayer with the same size."
        :param x:
        :param sublayer: 这里的子层是指 1 个 Encoder 层里面的一层，如 Multi-Head Attention
        :return:
        """

        return x + self.dropout(sublayer(self.norm(x)))
