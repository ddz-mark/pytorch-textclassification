# _*_coding:utf-8_*_
# user  : dudaizhong
# time  : 19-12-8 下午4:32
# info  :
import torch
import torch.nn.functional as F
from torch import nn

from model.base.Conv1d import Conv1d


class TextCNN(nn.Module):
    def __init__(self, embedding_dim, n_filters, filter_sizes, output_dim,
                 pretrained_embeddings, dropout=0.5):
        super(TextCNN, self).__init__()
        #         self.embedding = nn.Embedding(len_vocab,100)
        self.embedding = nn.Embedding.from_pretrained(
            pretrained_embeddings, freeze=False)

        self.convs = Conv1d(embedding_dim, n_filters, filter_sizes)

        self.fc = nn.Linear(len(filter_sizes) * n_filters, output_dim)
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        text, _ = x

        batch_size, seq_num = text.shape
        #         print(text.shape) #(batch_size,sent_len),(128,40)
        vec = self.embedding(text)
        #         print(vec.shape) #(batch_size,sent_len,emb_dim),(128,40,100)
        vec = vec.permute(0, 2, 1)
        #         print(vec.shape) #(batch_size,emb_dim,sent_len),(128,100,40)

        conved = self.convs(vec)
        #         print([conv.shape for conv in conved])
        #         (batch_size,n_filters,sent_len - filter_sizes[n] - 1)([128, 100, 40-2+1]), torch.Size([128, 100, 37]), torch.Size([128, 100, 36])

        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2)
                  for conv in conved]
        #         print([pool.shape for pool in pooled])
        #        (batch_size, n_filters)([128, 100]), torch.Size([128, 100]), torch.Size([128, 100])

        #         cat函数将（A,B），dim=0按行拼接，dim=1按列拼接
        cat = self.dropout(torch.cat(pooled, dim=1))
        #         print(cat.shape) # [128, 300]
        out = self.fc(cat)
        #         print(out.shape) # [128, 2]

        return out
