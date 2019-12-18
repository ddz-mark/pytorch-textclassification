# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
import random
import torch


def seed_everything(seed=2019):
    '''
    设置随机种子，最好在训练的时候调用
    '''
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_device():
    '''
    获取机器的cpu或者gpu
    '''
    device_all = []
    n_gpu = torch.cuda.device_count()
    for index in range(n_gpu):
        device_all.append(torch.device("cuda:"+str(index) if torch.cuda.is_available() else "cpu"))
        
    if torch.cuda.is_available():
        print("device is cuda, # cuda is: ", n_gpu)
    else:
        print("device is cpu, not recommend")
    return device_all, n_gpu


def seq_statistics(tokens):
    '''
    统计tokens长度，获取最优的 seq
    '''
    # 分词
#     tokens = nltk.word_tokenize(sentence)

    count_num_dict = {}

    length = len(tokens)
    print(length)
    if len(str(length)) == 1: # 1 位数 
        key = '10'
    elif len(str(length)) == 2: # 两位数
        key = str(str(length)[0]+'0')
    elif len(str(length)) == 3: # 三位数
        key = str(str(length)[0:2]+'0')
    else:
        print('tokens is so long')
        key = '>1000'
        
    if key in count_num_dict.keys():
        count_num_dict[key] += 1
    else:
        count_num_dict[key] =1
        
    return count_num_dict