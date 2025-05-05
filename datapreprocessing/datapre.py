#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 17 12:14:52 2018

@author: rajesh
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df=pd.read_csv("Data.csv")
X= df.iloc[:,:-1].values
y=df.iloc[:,3].values


from sklearn.impute import SimpleImputer
imputer=SimpleImputer(missing_values=np.nan,strategy="mean")
imputer.fit(X[:,1:3])
X[:,1:3]=imputer.transform(X[:,1:3])
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

# Assuming X is a NumPy array with mixed data types
columnTransformer = ColumnTransformer(
    transformers=[('encoder', OneHotEncoder(), [0])],
    remainder='passthrough'
)

X = columnTransformer.fit_transform(X)
#X = X.astype(str)  # Replace np.str with str


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)


from sklearn.preprocessing import StandardScaler
scx=StandardScaler()
X[:,3:5]=scx.fit_transform(X[:,3:5])