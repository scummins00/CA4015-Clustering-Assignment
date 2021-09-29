#!/usr/bin/env python
# coding: utf-8

# ## Lets read in some basic packages:

# In[1]:


import pandas as pd
import numpy as np
from sklearn.cluster import KMeans


# # Lets begin by reading in some data

# In[2]:


index_95 = pd.read_csv("../data/index_95.csv")
choice_95 = pd.read_csv("../data/choice_95.csv")
win_95 = pd.read_csv("../data/wi_95.csv")
loss_95 = pd.read_csv("../data/lo_95.csv")


# First thing I want to do is cluster by **Net Win $\times$ Net Loss** for each participant
# 
# Let's create a new DF with two columns, net_win & net_loss with each row representing a participant. We need:
# 
# 1. Array representing net win / loss for each participant
# 2. Suitable dataframe

# In[3]:


net_win = np.array(win_95.sum(axis=1))
net_loss = np.array(loss_95.sum(axis=1))
print(net_win + net_loss)


# In[ ]:




