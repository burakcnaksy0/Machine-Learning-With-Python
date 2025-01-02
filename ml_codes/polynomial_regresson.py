import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

data = pd.read_csv("C:/Users/burak/OneDrive/Masaüstü/ml/csv_files/poly.csv")

x  = data.zaman.values.reshape(-1,1)
y  = data.sicaklik.values.reshape(-1,1)
print("--x--")
print(x)
print("--y--")
print(y)
lr = LinearRegression()
lr.fit(x,y)

pr = PolynomialFeatures(degree=6)
xpl1 = pr.fit_transform(x)
lr2 = LinearRegression()
lr2.fit(xpl1,y)

plt.scatter(x, y)
plt.plot(x,lr.predict(x), color = "red")
plt.plot(x,lr2.predict(xpl1), color = "green")
plt.show()
