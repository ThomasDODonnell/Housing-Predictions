import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression 
from sklearn.metrics import mean_squared_error, mean_absolute_error 
from sklearn import preprocessing 

tdata = pd.read_csv("train.csv")
print(tdata.head())

print(tdata.columns)
print(tdata.BsmtCond)

bsmt_vals = tdata['BsmtCond'].value_counts()
print(bsmt_vals)
