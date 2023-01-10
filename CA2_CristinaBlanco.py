#!/usr/bin/env python
# coding: utf-8

# In[33]:


#Libraries
import pandas as pd
import numpy as np
from pathlib import Path #this library minimizes issues when using paths in Mac
import os
#pip install dataprep --user
from dataprep.eda import create_report
import re
import seaborn as sns
import streamlit as st
import datetime

import matplotlib.pyplot as plt

from warnings import filterwarnings
filterwarnings('ignore')

from scipy.stats.mstats import winsorize
import scipy.stats as stats

import statsmodels.api as sm
from statsmodels.tools.eval_measures import mse, rmse

from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error,explained_variance_score
from sklearn.linear_model import LinearRegression,Ridge,Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import (train_test_split,GridSearchCV, cross_val_score, cross_val_predict, StratifiedKFold, learning_curve)


# In[34]:


#pip freeze > requirements.txt


# In[35]:


#Seaborn Context
sns.set_theme(style='whitegrid', context='talk', palette='deep')
#plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1));
#sns.set_palette(['#62C370', '#FFD166', '#EF476F'])# Plot
#plt.figure(figsize=(9, 5)) 


# In[36]:


#List of Folders
prod_folder = Path("data/Value_of_Production_E_All_Data")


# In[37]:


prod_df = pd.read_csv(prod_folder /"Value_of_Production_E_All_Data.csv", encoding = 'unicode_escape')


# In[38]:


prod_df.head()


# In[39]:


#Remove F columns as they only contain "E" as index.
prod_df = prod_df[prod_df.columns.drop(list(prod_df.filter(regex='.*?F')))]
prod_df.head()


# In[40]:


prod_df = prod_df[prod_df["Element"]=='Gross Production Value (constant 2014-2016 thousand I$)']
prod_df = prod_df[prod_df.columns.drop(["Area Code (M49)", "Area Code", "Item Code","Item Code (CPC)", "Element", "Element Code", "Unit"])]


# In[41]:


prod_df


# In[42]:


prod_wide=pd.wide_to_long(prod_df, stubnames='Y', i=['Area', 'Item'], j='Year').reset_index()


# In[43]:


prod_wide


# In[44]:


prod_wide = prod_wide[prod_wide["Area"].isin(['Ireland', 'Colombia', 'New Zealand', 'Denmark', 'Spain'])]


# In[45]:


prod_wide = prod_wide[prod_wide["Item"].isin(['Apples', 'Hen eggs in shell, fresh', 'Meat indigenous, total', 'Milk, Total'])]


# In[46]:


create_report(prod_wide).show_browser()


# In[47]:


for item in prod_wide['Item'].unique():
    fig = plt.figure(figsize=(8,7))
    plt.title(item)
    sns.lineplot(data=prod_wide[prod_wide['Item']==item].drop('Item', axis=1), x='Year', y='Y', hue='Area')
    st.pyplot(fig)


# In[48]:


fig = plt.figure(figsize=(8,7))
sns.lineplot(data=prod_wide[prod_wide['Item']=='Milk, Total'].drop('Item', axis=1), x='Year', y='Y', hue='Area')
st.pyplot(fig)


# In[ ]:




