
# coding: utf-8

# # 导库

# In[1]:


import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

from torchtext import data
from torchtext import datasets
from torchtext import vocab

from tqdm import tqdm

import pandas as pd
import numpy as np
import random

import os

from sklearn.metrics import roc_auc_score

# 导入自己的库
from Util.utils import seed_everything,get_device
from Util.SST2_data import load_sst2
from ModelHandler import *


# In[5]:


device, n_gpu=get_device()
print(device, n_gpu)


# # 加载 SST2 数据

# In[3]:


# SST2 数据准备

text_field = data.Field(tokenize='spacy', lower=True, fix_length=40, batch_first=True)
label_field = data.LabelField(dtype=torch.long)


# In[4]:


BASE_PATH = "/home/dudaizhong/Downloads/SST-2/"
train_pd = pd.read_csv(BASE_PATH+'train.tsv', sep='\t')
dev_pd = pd.read_csv(BASE_PATH + 'dev.tsv', sep='\t')
test_pd = pd.read_csv(BASE_PATH + 'test.tsv', sep='\t')

print(train_pd.shape)
print(dev_pd.shape)
print(test_pd.shape)


# In[6]:


batch_size = 128
train_iter, dev_iter, test_iter = load_sst2(BASE_PATH, text_field, label_field, batch_size, device)


# # 网络结构

# In[7]:


# 1.与维度变换相关函数 view()，permute()，size(), torch.squeeze() / torch.unsqueeze()
# 2.Embedding层加载预训练模型的方式：1）copy，2）from_pretrained。

class Enet(nn.Module):
    def __init__(self,pretrained_embeddings):
        super(Enet, self).__init__()
#         self.embedding = nn.Embedding(len_vocab,100)
        self.embedding = nn.Embedding.from_pretrained(
            pretrained_embeddings, freeze=False)
        # LSTM 参数以及输入输出说明：
        # 结构参数：LSTM(input_size, hidden_size, num_layers)
        # input_size:输入的特征数量
        # hidden_size:隐藏的特征数量
        # num_layers:层数
        self.lstm = nn.LSTM(100,64,3,batch_first=True)#,bidirectional=True)
        self.linear = nn.Linear(64,2)
        
    def forward(self, x):
        batch_size,seq_num = x.shape
#         print(x.shape) #(128,40)
        vec = self.embedding(x)
#         print(vec.shape) #(128,40,100)
        out, (hn, cn) = self.lstm(vec)
#         print(out.shape) #(128,40,64)
        out = self.linear(out[:,-1,:])
#         print(out.shape) #(128,2)
        out = F.softmax(out,-1)
        return out
    


# # 训练验证

# In[8]:


get_ipython().run_cell_magic('time', '', "# seed_everything()\n\ntrain_batch_size, val_batch_size = 2**7, 2**7\n\npretrained_embeddings = text_field.vocab.vectors\nmodel = Enet(pretrained_embeddings)\n\n\nmodelHandlerParams = {}\nmodelHandlerParams['epoch_num'] = 1000000\nmodelHandlerParams['train_batch_size'] = train_batch_size\nmodelHandlerParams['val_batch_size'] = val_batch_size\nmodelHandlerParams['device'] = device\n\nmodelHandlerParams['model'] = model\nmodelHandler = ModelHandler(modelHandlerParams)\n\n# 二分类交叉熵\nloss_fn = nn.BCEWithLogitsLoss().to(device)\n# 调参地方，分别调整为0.1,0.01,0.001，最优为0.01\noptimizer = optim.Adam(model.parameters(), lr=0.01,\n                       weight_decay=0.00001) # lr sets the learning rate of the optimizer\n\nmodelHandler.fit(train_iter=train_iter, val_iter=dev_iter,loss_fn=loss_fn,optimizer=optimizer,\n                 early_stopping_rounds=10, verbose=2)")


# In[ ]:





# In[ ]:




