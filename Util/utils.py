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
    if n_gpu > 1:
        for index in range(n_gpu):
            device_all.append(torch.device("cuda:"+str(index) if torch.cuda.is_available() else "cpu"))
    else:
        device_all.append(torch.device("cpu"))
        
    if torch.cuda.is_available():
        print("device is cuda, # cuda is: ", n_gpu)
    else:
        print("device is cpu, not recommend")
    return device_all, n_gpu
