# _*_coding:utf-8_*_
# user  : dudaizhong
# time  : 19-12-10 下午8:29
# info  :
import math

from torch import nn


class Embeddings(nn.Module):
    def __init__(self, pretrained_embeddings, d_model, freeze=False):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding.from_pretrained(
            pretrained_embeddings, freeze=freeze)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)
