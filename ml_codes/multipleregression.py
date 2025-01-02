import numpy as np
import pandas as pd
import matplotlib as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

data = pd.read_csv("C:/Users/burak/OneDrive/Masaüstü/reklam.csv")
#print(data.to_string())

x= data.iloc[:,1:4].values
y= data.satış.values.reshape(-1,1)

xtrain,xtest,ytrain,ytest = train_test_split(x,y,train_size=0.7,random_state=33)
print("-----------ytest--------------")
print(ytest)
print("-----------ytrain--------------")
print(ytrain)
print("-------------------------")
lr = LinearRegression()
lr.fit(xtrain, ytrain)
yhead = lr.predict(xtest)
print(yhead)

print(lr.predict([[44.5,39.3,45.1]]))

