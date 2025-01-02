import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

data = pd.read_csv("C:/Users/burak/OneDrive/Masaüstü/ml/csv_files/ürün.csv")

x = data.iloc[:,0:2].values
y= data.satinalma.values.reshape(-1,1)

xtrain,xtest,ytrain,ytest = train_test_split(x,y,test_size=0.3,random_state=25)

s = data[data.satinalma==0]
b = data[data.satinalma==1]

plt.scatter(s.yaş,s.maaş,color ="red")
plt.scatter(b.yaş,b.maaş,color ="green")
plt.xlabel("Age")
plt.ylabel("Salary")
plt.show()