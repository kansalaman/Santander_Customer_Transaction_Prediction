
# coding: utf-8

# In[5]:

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler


# In[6]:

train=pd.read_csv("train.csv")
test=pd.read_csv("test.csv")


# In[7]:

train.head()


# In[8]:

test.head()


# In[9]:

train.set_index("ID_code",inplace=True)
test.set_index("ID_code",inplace=True)


# In[10]:

y_train=train["target"]
train.drop(labels="target",axis=1,inplace=True)
# a list should have been given instead of single element if
#you wanted to drop many columns
train.shape,test.shape


# In[11]:

scaler=MinMaxScaler()
X_train_scale=scaler.fit_transform(train)
X_test_scale=scaler.fit_transform(test)


# In[16]:

from sklearn.model_selection import train_test_split
X_train_sub, X_validation_sub, y_train_sub, y_validation_sub=train_test_split(X_train_scale,y_train,random_state=0,train_size=0.8)
X_train_sub.shape,X_validation_sub.shape


# In[18]:

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report,confusion_matrix,roc_curve,auc
learning_rates=[0.05,0.1,0.25,0.5,0.75,1]
for learning_rate in learning_rates:
    gb=GradientBoostingClassifier(n_estimators=200,
                                 learning_rate=learning_rate,max_features=2,
                                 max_depth=4,random_state=0)
    gb.fit(X_train_sub,y_train_sub)
    print("Learning rate: ", learning_rate)
    print("Accuracy score (training): {0:.3f}".format(gb.score(X_train_sub, y_train_sub)))
    print("Accuracy score (validation): {0:.3f}".format(gb.score(X_validation_sub, y_validation_sub)))


# In[27]:

gb=GradientBoostingClassifier(n_estimators=200,learning_rate=0.25,
                             max_features=2,max_depth=2,random_state=0)
gb.fit(X_train_sub,y_train_sub)

# you can actually see confusion matrix and classification report here 
#if you really want


# In[36]:

#You can also do here some ROC etc and other calculation
# here is the link https://www.kaggle.com/beagle01/prediction-with-gradient-boosting-classifier
# test_val=X_test_scale.values
predictions=gb.predict(X_test_scale)
X_test_scale.shape,predictions.sum()
arr=test.index
ans=pd.DataFrame({'ID_code':arr,'target':predictions})
ans.set_index('ID_code',inplace=True)
pd.DataFrame.to_csv(ans,'anss.csv')
