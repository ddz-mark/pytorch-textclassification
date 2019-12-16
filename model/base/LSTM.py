# _*_coding:utf-8_*_
# user  : dudaizhong
# time  : 19-12-8 下午3:17
# info  :

import torch.nn as nn
import torch.nn.functional as F
import torch


class LSTM(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers, bidirectional, batch_first=True, dropout=0):
        """
        Args:
            input_size: x 的特征维度
            hidden_size: 隐层的特征维度
            num_layers: LSTM 层数
        """
        super(LSTM, self).__init__()

        self.rnn = nn.LSTM(
            input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, bidirectional=bidirectional,
            batch_first=batch_first, dropout=dropout
        )

        self.init_params()

    def init_params(self):
        for i in range(self.rnn.num_layers):
            nn.init.orthogonal_(getattr(self.rnn, 'weight_hh_l{}'.format(i)))
            nn.init.kaiming_normal_(getattr(self.rnn, 'weight_ih_l{}'.format(i)))
            nn.init.constant_(getattr(self.rnn, 'bias_hh_l{}'.format(i)), val=0)
            nn.init.constant_(getattr(self.rnn, 'bias_ih_l{}'.format(i)), val=0)
            getattr(self.rnn, 'bias_hh_l{}'.format(i)).chunk(4)[1].fill_(1)

            if self.rnn.bidirectional:
                nn.init.orthogonal_(
                    getattr(self.rnn, 'weight_hh_l{}_reverse'.format(i)))
                nn.init.kaiming_normal_(
                    getattr(self.rnn, 'weight_ih_l{}_reverse'.format(i)))
                nn.init.constant_(
                    getattr(self.rnn, 'bias_hh_l{}_reverse'.format(i)), val=0)
                nn.init.constant_(
                    getattr(self.rnn, 'bias_ih_l{}_reverse'.format(i)), val=0)
                getattr(self.rnn, 'bias_hh_l{}_reverse'.format(i)).chunk(4)[1].fill_(1)

    def forward(self, x, lengths):
        # x: [seq_len, batch_size, input_size]
        # lengths: [batch_size]
        packed_x = nn.utils.rnn.pack_padded_sequence(x, lengths)

        # packed_x， packed_output: PackedSequence 对象
        # hidden: [num_layers * bidirectional, batch_size, hidden_size]
        # cell: [num_layers * bidirectional, batch_size, hidden_size]
        packed_output, (hidden, cell) = self.rnn(packed_x)

        # output: [real_seq_len, batch_size, hidden_size * 2]
        # output_lengths: [batch_size]
        output, output_lengths = nn.utils.rnn.pad_packed_sequence(packed_output)

        return output, hidden
