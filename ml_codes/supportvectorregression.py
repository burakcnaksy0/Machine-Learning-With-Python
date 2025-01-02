import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

data = pd.read_csv("C:/Users/burak/OneDrive/Masaüstü/ml/csv_files/poly.csv")

x  = data.zaman.values.reshape(-1,1)
y  = data.sicaklik.values.reshape(-1,1)

sc = StandardScaler()

x1 = sc.fit_transform(x)
y1 = sc.fit_transform(y)

sv = SVR(kernel="rbf")

sv.fit(x1,y1)

plt.scatter(x1,y1)
plt.plot(x1,sv.predict(x1), color ="red")
plt.show()