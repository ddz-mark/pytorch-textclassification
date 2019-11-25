
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
import random

import os

import torch


# In[4]:


def seed_everything(seed=2019):
    '''
    设置随机种子，最好在训练的时候调用
    '''
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    
def get_device():
    '''
    获取机器的cpu或者gpu
    '''
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    if torch.cuda.is_available():
        print("device is cuda, # cuda is: ", n_gpu)
    else:
        print("device is cpu, not recommend")
    return device, n_gpu


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:



