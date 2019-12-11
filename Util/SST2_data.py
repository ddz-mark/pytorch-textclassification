# coding: utf-8

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
from Util.utils import seed_everything, get_device
from ModelHandler import *


def load_sst2(path, text_field, label_field, batch_size, device):
    # 2. 定义 DataSet
    train, dev = data.TabularDataset.splits(
        path=path, train='train.tsv', validation='dev.tsv', format='tsv', skip_header=True,
        fields=[('text', text_field), ('label', label_field)])

    # 这里需要注意单独处理的时候不能用 splits 方法。
    test = data.TabularDataset(path + 'test.tsv', format='tsv', skip_header=True,
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
    #    for batch_idx, (X_train_var, y_train_var) in enumerate(train_iter):
    #        print("the shape of train_x: {}, train_y:{}".format(X_train_var.shape, y_train_var.shape))
    #        break

    return train_iter, dev_iter, test_iter
