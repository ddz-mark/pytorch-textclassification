# _*_coding:utf-8_*_
# user  : dudaizhong
# time  : 19-12-8 下午4:35
# info  :
# 卷积网络一般情况
import torch

from torch import nn

from model.base.LSTM import LSTM


class TextRNN(nn.Module):
    def __init__(self, embedding_dim, output_dim, hidden_size, num_layers, bidirectional, pretrained_embeddings, dropout=0.5):
        super(TextRNN, self).__init__()

        self.hidden_size = hidden_size

        #         self.embedding = nn.Embedding(len_vocab,100)
        self.embedding = nn.Embedding.from_pretrained(
            pretrained_embeddings, freeze=False)
        # LSTM 参数以及输入输出说明：
        # 结构参数：LSTM(input_size, hidden_size, num_layers)
        # input_size:输入的特征数量
        # hidden_size:隐藏的特征数量
        # num_layers:层数
        self.lstm = LSTM(embedding_dim, hidden_size, num_layers, bidirectional=bidirectional)  # ,bidirectional=True)

        self.fc = nn.Linear(hidden_size * 2, output_dim)
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        text, text_lengths = x

        batch_size, sent_len = text.shape
        #         print(text.shape) #(batch_size 128, sent_len 40)
        vec = self.dropout(self.embedding(text))
        #         print(vec.shape) #(batch_size 128,sent_len 40,emb_dim 100)
        vec = vec.permute(1, 0, 2)
        lstm_out, hn = self.lstm(vec, text_lengths)
        # print(lstm_out.shape)  # (sent_len 40,batch_size 128,hidden_size*2 128)
        # 这里进行前后连接时，使用的隐藏状态 hn 的最后一层 与 直接使用lstm_out中最后一层有不一样
        out = self.dropout(torch.cat((hn[-2, :, :], hn[-1, :, :]), dim=1))
        # print(out.shape) # (batch_size 128, sent_len 100)
        out = self.fc(out)
        # print(out.shape)
        return out
