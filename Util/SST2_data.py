
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
from ModelHandler import *


# In[2]:


device, n_gpu = get_device()
print(device)


# # 数据预处理

# In[3]:


BASE_PATH = "/home/dudaizhong/Downloads/SST-2/"
train_pd = pd.read_csv(BASE_PATH+'train.tsv', sep='\t')
dev_pd = pd.read_csv(BASE_PATH + 'dev.tsv', sep='\t')
test_pd = pd.read_csv(BASE_PATH + 'test.tsv', sep='\t')

print(train_pd.shape)
print(dev_pd.shape)
print(test_pd.shape)


# ## 定义 Field

# In[4]:


# 1. 定义 Field

text_field = data.Field(tokenize='spacy', lower=True, fix_length=40, batch_first=True)
label_field = data.LabelField(dtype=torch.long)


# ## 定义 DataSet

# In[11]:


# 2. 定义 DataSet

train, dev = data.TabularDataset.splits(
        path=BASE_PATH, train='train.tsv', validation='dev.tsv',format='tsv', skip_header=True,
        fields=[('text', text_field), ('label', label_field)])

# 这里需要注意单独处理的时候不能用 splits 方法。
test = data.TabularDataset(BASE_PATH+'test.tsv', format='tsv', skip_header=True,
        fields=[('index', label_field), ('text', text_field)])

print("the size of train: {}, dev:{}, test:{}".format(
    len(train), len(dev), len(test)))


# In[17]:


# 查看 Example
print(train[1].text, train[1].label)

print(dev[1].text, dev[1].label)

print(test[1].text)


# ## 建立 Vocab

# In[18]:


# 3. 建立 vocab，大小是text_field里面的词数量
# vectors = vocab.Vectors(embedding_file, cache_dir)

text_field.build_vocab(
        train, dev, test, max_size=25000,
        vectors='glove.6B.100d', unk_init=torch.Tensor.normal_)

label_field.build_vocab(train, dev, test)


# In[19]:


len_vocab = len(text_field.vocab)
print(len_vocab)

print(len(label_field.vocab))
# for step, batch in enumerate(tqdm(train_iter, desc="Iteration")):
#     print(batch.text, batch.label)
    


# ## 构造 Iterater

# In[23]:


# 4. 构造迭代器

train_iter, dev_iter = data.BucketIterator.splits(
        (train, dev), batch_sizes=(128, 128), sort_key=lambda x: len(x.text), 
        sort_within_batch=True, repeat=False, shuffle=True, device=device)

# 同样单独处理的时候
test_iter = data.Iterator(test, batch_size=len(test), train=False,
                          sort=False, device=device)

print("the size of train_iter: {}, dev_iter:{}, test_iter:{}".format(
    len(train_iter), len(dev_iter), len(test_iter)))



# In[24]:


# 查看 Iterater
# seed_everything()
for batch_idx, (X_train_var, y_train_var) in enumerate(train_iter):
    print(batch_idx, X_train_var.shape, y_train_var.shape)
    break


# # 函数定义

# In[30]:


def load_sst2(path, text_field, label_field, batch_size, device):
    
    # 2. 定义 DataSet
    train, dev = data.TabularDataset.splits(
            path=path, train='train.tsv', validation='dev.tsv',format='tsv', skip_header=True,
            fields=[('text', text_field), ('label', label_field)])

    # 这里需要注意单独处理的时候不能用 splits 方法。
    test = data.TabularDataset(BASE_PATH+'test.tsv', format='tsv', skip_header=True,
            fields=[('index', label_field), ('text', text_field)])
    print("the size of train: {}, dev:{}, test:{}".format(len(train), len(dev), len(test)))
    print("the result of dataset: ", train[0].text, train[0].label)
    
    # 3. 建立 vocab，大小是text_field里面的词数量
    # vectors = vocab.Vectors(embedding_file, cache_dir)

    text_field.build_vocab(
            train, dev, test, max_size=25000,
            vectors='glove.6B.100d', unk_init=torch.Tensor.normal_)

    label_field.build_vocab(train, dev, test)
    
    # 4. 构造迭代器
    train_iter, dev_iter = data.BucketIterator.splits(
            (train, dev), batch_sizes=(batch_size, batch_size), sort_key=lambda x: len(x.text), 
            sort_within_batch=True, repeat=False, shuffle=True, device=device)

    # 同样单独处理的时候
    test_iter = data.Iterator(test, batch_size=len(test), train=False,
                              sort=False, device=device)

    print("the size of train_iter: {}, dev_iter:{}, test_iter:{}".format(
        len(train_iter), len(dev_iter), len(test_iter)))
    for batch_idx, (X_train_var, y_train_var) in enumerate(train_iter):
        print("the shape of train_x: {}, train_y:{}".format(X_train_var.shape, y_train_var.shape))
        break
    
    return train_iter, dev_iter, test_iter
    


# In[31]:


# 加载 SST2 数据集
batch_size = 128
train_iter, dev_iter, test_iter = load_sst2(BASE_PATH, text_field, label_field, batch_size, device)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




