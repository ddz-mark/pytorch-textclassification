# _*_coding:utf-8_*_
# user  : dudaizhong
# time  : 19-12-8 下午3:16
# info  :

import torch.nn as nn
import torch.nn.functional as F
import torch

from model.base.LSTM import LSTM


class TextRNN_Attention(nn.Module):
    def __init__(self, embedding_dim, output_dim, hidden_size, num_layers, bidirectional, pretrained_embeddings,
                 device):
        super(TextRNN_Attention, self).__init__()

        self.hidden_size = hidden_size
        self.device = device

        # self.embedding = nn.Embedding(len_vocab,100)
        self.embedding = nn.Embedding.from_pretrained(
            pretrained_embeddings, freeze=False)

        # LSTM 参数以及输入输出说明：
        # 结构参数：LSTM(input_size, hidden_size, num_layers)
        # input_size:输入的特征数量
        # hidden_size:隐藏的特征数量
        # num_layers:层数
        self.lstm = LSTM(embedding_dim, hidden_size, num_layers, bidirectional=bidirectional)  # ,bidirectional=True)

        self.fc = nn.Linear(hidden_size * 2, output_dim)

        # Attention 机制
        ws = torch.empty(self.hidden_size * 2, self.hidden_size * 2,
                         dtype=torch.float32, device=self.device,
                         requires_grad=True)
        nn.init.normal_(ws)

        # attention 机制，shape(self.hidden_size * 2)
        bw = torch.empty(self.hidden_size * 2,
                         dtype=torch.float32, device=self.device,
                         requires_grad=True)
        nn.init.normal_(bw)

        # attention 机制，shape(self.hidden_size * 2)
        us = torch.empty(self.hidden_size * 2,
                         dtype=torch.float32, device=self.device,
                         requires_grad=True)
        nn.init.normal_(us)

        self.ws = nn.Parameter(ws)
        self.bw = nn.Parameter(bw)
        self.us = nn.Parameter(us)

    def attention(self, ws, hi, bw, us):
        """
        根据 HAN 模型公式为：
        ui = tanh(ws * hi + bw)
        ai = softmax(ui(T) * us)
        s = 求和(ai * hi)
        :param ws: shape(hidden_size*2, hidden_size*2)
        :param hi: shape(batch_size, sent_len, hidden_size*2)
        :param bw: shape(hidden_size*2, 1)
        :param us: shape(hidden_size*2, 1)
        :return: s: shape(batch_size, sent_len, hidden_size*2)
        """
        ui = torch.tanh(torch.einsum("ble,ee->ble", [hi, ws]) + bw)
        #         print(torch.einsum("ble,e->bl", [ui, us]).shape)
        ai = F.softmax(torch.einsum("ble,e->bl", [ui, us]), dim=1)
        s = torch.einsum("ble,bl->ble", [hi, ai])
        s = torch.sum(s, dim=1)
        return s

    def forward(self, x):
        text, text_lengths = x

        batch_size, sent_len = text.shape
        # print(text.shape) #(batch_size 128, sent_len 40)
        vec = self.embedding(text)
        # print(vec.shape) #(batch_size 128,sent_len 40,emb_dim 100)
        vec = vec.permute(1, 0, 2)
        lstm_out, hn = self.lstm(vec, text_lengths)
        #         print(lstm_out.shape) #(real_sent_len 40,batch_size 128,hidden_size*2 128)
        lstm_out = lstm_out.permute(1, 0, 2)
        #         print("lstm_out: ", lstm_out.shape)  # (batch_size 128,real_sent_len,hidden_size*2 128)

        out = self.attention(self.ws, lstm_out, self.bw, self.us)
        out = self.fc(out)
        # print(out.shape)
        return out
