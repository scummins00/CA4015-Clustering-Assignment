#!/usr/bin/env python
# coding: utf-8

# # Data Cleaning
# In the following Notebook, we will verify the integrity of our data. The data provided by 10 individual studies and centralised by {cite}`Steingroever2015`, is inherently clean and ready for use. To ensure this, we will perform the following verification steps:
# 
# 1. Test all datasets for any missing values.
# 2. Verify that deck choice datasets do not host cells exceding a maximum value of 4 and a minimum value of 1.

# In[1]:


#importing packages
import pandas as pd
import numpy as np


# In[2]:


#Reading in the data
sets = []

#choices
choice_95=pd.read_csv("../data/choice_95.csv")
sets.append(choice_95)
choice_100=pd.read_csv("../data/choice_100.csv")
sets.append(choice_100)
choice_150=pd.read_csv("../data/choice_150.csv")
sets.append(choice_150)

#Losses
loss_95=pd.read_csv("../data/lo_95.csv")
sets.append(loss_95)
loss_100=pd.read_csv("../data/lo_100.csv")
sets.append(loss_100)
loss_150=pd.read_csv("../data/lo_150.csv")
sets.append(loss_150)
#Wins
win_95=pd.read_csv("../data/wi_95.csv")
sets.append(win_95)
win_100=pd.read_csv("../data/wi_100.csv")
sets.append(win_100)
win_150=pd.read_csv("../data/wi_150.csv")
sets.append(win_150)

#Index
index_95=pd.read_csv("../data/index_95.csv")
sets.append(index_95)
index_100=pd.read_csv("../data/index_100.csv")
sets.append(index_100)
index_150=pd.read_csv("../data/index_150.csv")
sets.append(index_150)

for set in sets:
    print("NaN value detected: {}".format(set.isnull().values.any()))


# In[3]:


#Let's view the statitics for our choice datasets to verify min & max values
(choice_100.T.describe()).iloc[[3,-1]].mean(axis=1)


# **Note: This part of data exploration is simple by nature, but generates a large output. For simplicity, I have only included the verification of the small choice_100 dataset.**

# # Data Preperation
# In section 4 of this book, we will be performing K-Means clustering based on a participant's cumulative win and loss over the course of the game with results measured in 10% intervals of completion. That is to say, **we measure a participants net score every 10 turns in the case of a participant with 100 total turns**.
# 
# In the case of participants with 150 attempts, every **15 consecutive attempts will be condensed into a singular value**.
# 
# In the case of participants with 95 attempts, some calculation will be required to aggregate data points together and obtain a mean value. This is required as *10% of 95 is 9.5*. Clearly we cannot measure the 9.5th turn. This means we will measure **the mean of the 9th and 10th turn**.
# 
# To do this, our data requires some **Feature Engineering**. We require a new dataset consisting of the scores described above *per participant*. Also, in Section 5, we will be performing the same analysis, but with a **Federated Learning** approach. This means that one large dataset will not suffice. For each of the original datasets provided we must:
# 
# 1. Create and fill our rolling score datasets
# 2. Divide the data out into their individual surveys 

# ## Creating Rolling Dataframes
# The creation of Dataframes to hold rolling cumulative sumations of values across periods of 10 & 15 attempts for the surveys allowing 100 & 150 attempts respectfully is a painless process.
# 
# However, this is not the case with the survey offering 95 attempts as 95 is an uneven number meaning it does not divide easily into equally sized portions. As a consequence of this, the processing steps for the 95 dataset are much more complex.

# In[4]:


#We will use pandas.DataFrame.cumsum() to calculate our cumulative sum

rolling_win_100=(win_100.cumsum(axis=1)).iloc[:, range(9,100,10)]
rolling_loss_100=(loss_100.cumsum(axis=1)).iloc[:, range(9,100,10)]

rolling_win_150=(win_150.cumsum(axis=1)).iloc[:, range(14,150,15)]
rolling_loss_150=(loss_150.cumsum(axis=1)).iloc[:, range(14,150,15)]


# In[5]:


#The rolling values for the 95 sets are more difficult as 95 is not divisible by 10
inter_95=(win_95.cumsum(axis=1)).iloc[:,[9,18,27,28,37,46,47,56,65,66,75,84,85, 94]]

#Finding the rolling sum for 9th column
wins_95_col8=(win_95.cumsum(axis=1)).iloc[:,8]

#Calculating the average of intermediate columns as new column
Wins_9_5=(wins_95_col8+inter_95.iloc[:,0])/2
Wins_28_5=(inter_95.iloc[:,2]+inter_95.iloc[:,3])/2
Wins_47_5=(inter_95.iloc[:,5]+inter_95.iloc[:,6])/2
Wins_66_5=(inter_95.iloc[:,8]+inter_95.iloc[:,9])/2
Wins_85_5=(inter_95.iloc[:,11]+inter_95.iloc[:,12])/2

#Add everything together
inter_win_95=pd.concat([inter_95, Wins_9_5.rename("Wins_9_5"), Wins_28_5.rename("Wins_28_5"), 
                        Wins_47_5.rename("Wins_47_5"),Wins_66_5.rename("Wins_66_5"),
                        Wins_85_5.rename("Wins_85_5")], axis=1)

#Reorganise columns
cols = inter_win_95.columns.tolist()
rolling_win_95 = inter_win_95[[cols[-5], cols[1], cols[-4], cols[4], cols[-3], cols[7], cols[-2], cols[10], cols[-1], cols[13]]]


# In[6]:


#Now we must do the same for the Losses
#The rolling values for the 95 sets are more difficult as 95 is not divisible by 10
inter_loss_95=(loss_95.cumsum(axis=1)).iloc[:,[9,18,27,28,37,46,47,56,65,66,75,84,85, 94]]

#Finding the rolling sum for 9th column
losses_95_col8=(loss_95.cumsum(axis=1)).iloc[:,8]

#Calculating the average of intermediate columns as new column
Losses_9_5=(losses_95_col8+inter_loss_95.iloc[:,0])/2
Losses28_5=(inter_loss_95.iloc[:,2]+inter_loss_95.iloc[:,3])/2
Losses47_5=(inter_loss_95.iloc[:,5]+inter_loss_95.iloc[:,6])/2
Losses66_5=(inter_loss_95.iloc[:,8]+inter_loss_95.iloc[:,9])/2
Losses85_5=(inter_loss_95.iloc[:,11]+inter_loss_95.iloc[:,12])/2

#Add everything together
inter_loss_95=pd.concat([inter_loss_95, Losses_9_5.rename("Losses_9_5"), Losses28_5.rename("Losses28_5"), 
                        Losses47_5.rename("Losses47_5"),Losses66_5.rename("Losses66_5"),
                        Losses85_5.rename("Losses85_5")], axis=1)

#Reorganise columns
cols = inter_loss_95.columns.tolist()
rolling_loss_95 = inter_loss_95[[cols[-5], cols[1], cols[-4], cols[4], cols[-3], cols[7], cols[-2], cols[10], cols[-1], cols[13]]]


# ## Seperate Data by Study
# We will now seperate our data by study. We can achieve this by using our `index` files which allows us to seperate our subjects row-wise. We will do the following:
# 
# 1. Append our index value as a new column
# 2. Group our data by this new column
# 3. Select each study as a subset and create a new DataFrame

# In[7]:


#List for sub sets
finished_sets = []

#List for full sets
full_sets = []


# ### Larger Rolling Datasets
# While the seperated studies will be beneficial for the federated learning approach, it is important to keep the aggregated datasets also.

# In[8]:


#Wins
full_rolling_wins_95 =rolling_win_95.reset_index(drop=True).join(index_95)
full_rolling_wins_100 =rolling_win_100.reset_index(drop=True).join(index_100)
full_rolling_wins_150 =rolling_win_150.reset_index(drop=True).join(index_150)
full_sets.append(full_rolling_wins_95)
full_sets.append(full_rolling_wins_100)
full_sets.append(full_rolling_wins_150)

#losses
full_rolling_losses_95=rolling_loss_95.reset_index(drop=True).join(index_95)
full_rolling_losses_100=rolling_loss_100.reset_index(drop=True).join(index_100)
full_rolling_losses_150=rolling_loss_150.reset_index(drop=True).join(index_150)
full_sets.append(full_rolling_losses_95)
full_sets.append(full_rolling_losses_100)
full_sets.append(full_rolling_losses_150)

#Choices
full_rolling_choices_95=choice_95.reset_index(drop=True).join(index_95)
full_rolling_choices_100=choice_100.reset_index(drop=True).join(index_100)
full_rolling_choices_150=choice_150.reset_index(drop=True).join(index_150)
full_sets.append(full_rolling_choices_95)
full_sets.append(full_rolling_choices_100)
full_sets.append(full_rolling_choices_150)


# ### The 95 Dataset:

# In[9]:


#Wins
Fridberg_rolling_wins_95=rolling_win_95.reset_index(drop=True).join(index_95)
finished_sets.append(Fridberg_rolling_wins_95)

#Losses
Fridberg_rolling_losses_95=rolling_loss_95.reset_index(drop=True).join(index_95)
finished_sets.append(Fridberg_rolling_losses_95)

#Choices
Fridberg_choices_95=choice_95.reset_index(drop=True).join(index_95)
finished_sets.append(Fridberg_choices_95)


# ### The 100 dataset:

# In[10]:


#Wins
grouped_wins_100 = rolling_win_100.reset_index(drop=True).join(index_100).groupby("Study")

Horstmann_rolling_wins_100=grouped_wins_100.get_group("Horstmann")
finished_sets.append(Horstmann_rolling_wins_100)

Kjome_rolling_wins_100=grouped_wins_100.get_group("Kjome")
finished_sets.append(Kjome_rolling_wins_100)

Maia_rolling_wins_100=grouped_wins_100.get_group("Maia")
finished_sets.append(Maia_rolling_wins_100)

SteingroverInPrep_rolling_wins_100=grouped_wins_100.get_group("SteingroverInPrep")
finished_sets.append(SteingroverInPrep_rolling_wins_100)

Premkumar_rolling_wins_100=grouped_wins_100.get_group("Premkumar")
finished_sets.append(Premkumar_rolling_wins_100)

Wood_rolling_wins_100=grouped_wins_100.get_group("Wood")
finished_sets.append(Wood_rolling_wins_100)

Worthy_rolling_wins_100=grouped_wins_100.get_group("Worthy")
finished_sets.append(Worthy_rolling_wins_100)


# In[11]:


#Losses
grouped_losses_100 = rolling_loss_100.reset_index(drop=True).join(index_100).groupby("Study")

Horstmann_rolling_losses_100=grouped_losses_100.get_group("Horstmann")
finished_sets.append(Horstmann_rolling_losses_100)

Kjome_rolling_losses_100=grouped_losses_100.get_group("Kjome")
finished_sets.append(Kjome_rolling_losses_100)

Maia_rolling_losses_100=grouped_losses_100.get_group("Maia")
finished_sets.append(Maia_rolling_losses_100)

SteingroverInPrep_rolling_losses_100=grouped_losses_100.get_group("SteingroverInPrep")
finished_sets.append(SteingroverInPrep_rolling_losses_100)

Premkumar_rolling_losses_100=grouped_losses_100.get_group("Premkumar")
finished_sets.append(Premkumar_rolling_losses_100)

Wood_rolling_losses_100=grouped_losses_100.get_group("Wood")
finished_sets.append(Wood_rolling_losses_100)

Worthy_rolling_losses_100=grouped_losses_100.get_group("Worthy")
finished_sets.append(Worthy_rolling_losses_100)


# In[12]:


#Choices
grouped_choices_100 = choice_100.reset_index(drop=True).join(index_100).groupby("Study")

Horstmann_choices_100=grouped_choices_100.get_group("Horstmann")
finished_sets.append(Horstmann_choices_100)

Kjome_choices_100=grouped_choices_100.get_group("Kjome")
finished_sets.append(Kjome_choices_100)

Maia_choices_100=grouped_choices_100.get_group("Maia")
finished_sets.append(Maia_choices_100)

SteingroverInPrep_choices_100=grouped_choices_100.get_group("SteingroverInPrep")
finished_sets.append(SteingroverInPrep_choices_100)

Premkuma_choices_100=grouped_choices_100.get_group("Premkumar")
finished_sets.append(Premkuma_choices_100)

Wood_choices_100=grouped_choices_100.get_group("Wood")
finished_sets.append(Wood_choices_100)

Worthy_choices_100=grouped_choices_100.get_group("Worthy")
finished_sets.append(Worthy_choices_100)


# ### The 150 Dataset:

# In[13]:


#Wins
grouped_wins_150 = rolling_win_150.reset_index(drop=True).join(index_150).groupby("Study")

Steingroever2011_rolling_wins_150=grouped_wins_150.get_group("Steingroever2011")
finished_sets.append(Steingroever2011_rolling_wins_150)
Wetzels_rolling_wins_150=grouped_wins_150.get_group("Wetzels")
finished_sets.append(Wetzels_rolling_wins_150)


# In[14]:


#Losses
grouped_losses_150 = rolling_loss_150.reset_index(drop=True).join(index_150).groupby("Study")

Steingroever2011_rolling_losses_150=grouped_losses_150.get_group("Steingroever2011")
finished_sets.append(Steingroever2011_rolling_losses_150)
Wetzels_rolling_losses_150=grouped_losses_150.get_group("Wetzels")
finished_sets.append(Wetzels_rolling_losses_150)


# In[15]:


#Choices
grouped_choices_150 = choice_150.reset_index(drop=True).join(index_150).groupby("Study")

Steingroever2011_choices_150=grouped_choices_150.get_group("Steingroever2011")
finished_sets.append(Steingroever2011_choices_150)
Wetzels_choices_150=grouped_choices_150.get_group("Wetzels")
finished_sets.append(Wetzels_choices_150)


# ### Writing Out Data:

# In[16]:


#Writing out full datasets
for s in full_sets:
    s.to_csv(f'../data/cleaned/full_{s.columns[0].split("_")[0]}_{s.columns[-3].split("_")[-1]}.csv', index=False)


# In[17]:


#Writing out study datasets
for s in finished_sets:
    s.to_csv(f'../data/cleaned/{s.Study.unique()[0]}_rolling_{s.columns[0].split("_")[0]}_{s.columns[-3].split("_")[-1]}.csv', index=False)


# ## Conclusion
# We have now created a total of 39 datasets which are stored in a folder called `cleaned`. These sets consist of 9 full sets describing the amounts participants made and lost over 10% intervals. The other 30 sets are subsets of the larger 9 sets seperated by study.
# 
# We will use the larger sets in Section 4 of this book for K-Means Clustering and analysis. The subsets will then be used in Section 5 as part of a Federated Learning approach.
