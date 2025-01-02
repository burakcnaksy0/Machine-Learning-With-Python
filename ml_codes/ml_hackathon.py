import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import locale

# Türkçe ay adları için locale ayarı
locale.setlocale(locale.LC_TIME, "tr_TR.UTF-8")

# Excel dosyasını okuyalım
data = pd.read_excel("C:/Users/burak/OneDrive/Masaüstü/book.xlsx")

data["Tarih"] = pd.to_datetime(data["Tarih"], errors='coerce')
data["Tarih"] = data["Tarih"].fillna(pd.Timestamp('1970-01-01'))  # Use a default date for invalid entries
data["Tarih"] = data["Tarih"].map(pd.Timestamp.toordinal)


# Özellik ve hedef değişkenleri tanımlayalım
x = data["Tarih"].values.reshape(-1, 1)  # 2D array
y = data["Kapanış"].values  # 1D array

# Verileri görselleştirelim
plt.scatter(x, y)
plt.xlabel("Tarih (Ordinal)")
plt.ylabel("Kapanış")
plt.title("Tarih vs Kapanış")
plt.show()

# Verileri eğitim ve test setlerine bölelim
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.3, random_state=33)

# Modeli eğitelim
lr = LinearRegression()
lr.fit(xtrain, ytrain)

# Test verilerinde tahmin yapalım
yhead = lr.predict(xtest)

# Tahminleri ve gerçek değerleri gösterelim
print("Tahmin Edilen Kapanışlar:", yhead)
print("Gerçek Kapanışlar:", ytest)

# Verileri ve regresyon doğrusunu görselleştirelim
plt.scatter(x, y, label="Veriler")
plt.plot(xtest, lr.predict(xtest), color='red', label="Regresyon Çizgisi")
plt.xlabel("Tarih (Ordinal)")
plt.ylabel("Kapanış")
plt.legend()
plt.show()
