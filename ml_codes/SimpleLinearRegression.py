import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

data= pd.read_csv("C:/Users/burak/OneDrive/Masaüstü/ML/csv_files/satislar.csv")

print(data)

aylar = data[['Aylar']]
satıslar = data[['Satislar']]

print(aylar)
print(satıslar)

satıslar2 = data.iloc[:,-1].values
print(satıslar2)

x_train,x_test,y_train,y_test  = train_test_split(aylar,satıslar,test_size=0.33,random_state=1)

'''
sc= StandardScaler()    # verileri aynı dünyaya indirgemek
X_train = sc.fit_transform(x_train)
X_test = sc.fit_transform(x_test)
Y_train = sc.fit_transform(y_train)
Y_test = sc.fit_transform(y_test)

'''

lr = LinearRegression()
lr.fit(x_train,y_train)

predict = lr.predict(x_test)

x_train = x_train.sort_index()
y_train = y_train.sort_index()

plt.plot(x_train, y_train)
plt.plot(x_test,lr.predict(x_test))
plt.show()
























