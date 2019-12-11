# _*_coding:utf-8_*_
# user  : dudaizhong
# time  : 19-12-11 上午11:33
# info  :
from torch import nn

from model.transformer.Embeddings import Embeddings
from model.transformer.Encoder import Encoder
from model.transformer.EncoderLayer import EncoderLayer
from model.transformer.MultiHeadedAttention import MultiHeadedAttention
from model.transformer.PositionalEncoding import PositionalEncoding
from model.transformer.PositionwiseFeedForward import PositionwiseFeedForward


class Transformer_TextClassifier(nn.Module):
    """ 用 Transformer 来作为特征抽取的基本单元 """

    def __init__(self, head, n_layer, emd_dim, d_model, d_ff, output_dim, dropout, pretrained_embeddings):
        super(Transformer_TextClassifier, self).__init__()

        self.word_embedding = Embeddings(pretrained_embeddings, emd_dim)
        self.position_embedding = PositionalEncoding(emd_dim, dropout)

        # 这一层主要是调整维度，也可以放在最后的全连接层
        self.trans_linear = nn.Linear(emd_dim, d_model)

        multi_attn = MultiHeadedAttention(head, d_model)
        feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)

        self.encoder = Encoder(EncoderLayer(d_model, multi_attn, feed_forward, dropout), n_layer)
        self.fc = nn.Linear(d_model, output_dim)

    def forward(self, x):
        """
        x:
            text: [sent len, batch size], 文本数据
            text_lens: [batch_size], 文本数据长度
        """
        text, _ = x
        # text: [batch_size, sent_len]

        embeddings = self.word_embedding(text)
        # embeddings: [batch_size, sent_len, emd_dim]
        embeddings = self.position_embedding(embeddings)
        # embeddings: [batch_size, sent_len, emd_dim]

        embeddings = self.trans_linear(embeddings)
        # embeddings: [batch_size, sent_len, d_model]

        embeddings = self.encoder(embeddings)
        # embeddings: [batch_size, sent_len, d_model]

        features = embeddings[:, -1, :]
        # features: [batch_size, d_model]

        return self.fc(features)


