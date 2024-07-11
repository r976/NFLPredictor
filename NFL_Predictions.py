#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
import xgboost as xgb

# df2 = pd.read_csv('../ML/data/NFLv4.csv')
# df3 = pd.read_csv('../ML/data/NFLv5.csv')





# In[3]:


df = pd.read_csv('../ML/data/NFLv4.csv', low_memory=False)


# In[105]:


pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
# le = preprocessing.LabelEncoder()
# df['PlayType'] = le.fit_transform(df['PlayType'])
# df.corr().PlayType.sort_values(axis=0, ascending=True)
list = ['RushAttempt', 'Timeout_Indicator', 'Sack', 'PlayTimeDiff', 'Touchdown_Prob', 'yrdline100', 'PassAttempt', 'ExPoint_Prob', 'sp', 'Reception', 'down', 'PlayType']


# In[116]:


df_1 = df[list]
df_1 = df_1.dropna()
le = preprocessing.LabelEncoder()
X = df_1.drop('PlayType', axis=1)
y = df_1['PlayType']
y = le.fit_transform(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)


# In[107]:


dt = tree.DecisionTreeClassifier()
dt.fit(X_train, y_train)
y_pred = dt.predict(X_test)
accuracy_score(y_test, y_pred)


# In[113]:


gb = GradientBoostingClassifier(learning_rate=0.1)
gb.fit(X_train, y_train)
y_pred = gb.predict(X_test)
accuracy_score(y_test, y_pred)


# In[109]:


knn = KNeighborsClassifier(n_neighbors=10)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
accuracy_score(y_test, y_pred)


# In[110]:


rf = RandomForestClassifier(n_estimators=100)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
accuracy_score(y_test, y_pred)


# In[111]:


lr = LinearRegression()
lr.fit(X_train, y_train)
lr.score(X_test, y_test)


# In[112]:


# y = df_1['PlayType']
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

xg = xgb.XGBClassifier()
xg.fit(X_train, y_train)
y_pred = xg.predict(X_test)
accuracy_score(y_test, y_pred)

