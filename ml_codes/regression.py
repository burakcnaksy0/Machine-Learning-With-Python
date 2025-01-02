import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Veri dosyasını okuyalım
data = pd.read_csv("C:/Users/burak/OneDrive/Masaüstü/note.csv")

# Özellik ve hedef değişkenleri tanımlayalım
x = data["experience"].values.reshape(-1, 1)  # 2D array
y = data["maas"].values  # 1D array

# Verileri görselleştirelim
plt.scatter(x, y)
plt.xlabel("Experience")
plt.ylabel("Salary")
plt.title("Experience vs Salary")
#plt.show()

# Verileri eğitim ve test setlerine bölelim
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.3, random_state=33)

# Modeli eğitelim
lr = LinearRegression()
lr.fit(xtrain, ytrain)

# Test verilerinde tahmin yapalım
yhead = lr.predict(xtest)

# Tahminleri ve gerçek değerleri gösterelim
print("Tahmin Edilen Maaşlar:", yhead)
print("Gerçek Maaşlar:", ytest)
print(lr.predict([[10]]))

plt.scatter(x,y)
plt.plot(xtest,lr.predict(xtest),color='red')
plt.show()