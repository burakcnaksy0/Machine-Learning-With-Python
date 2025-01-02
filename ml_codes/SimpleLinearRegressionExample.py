import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Veriyi oku
data = pd.read_csv("C:/Users/burak/OneDrive/Masaüstü/ML/csv_files/ev_fiyatlari.csv")

# Sütun isimlerindeki boşlukları temizle
data.columns = data.columns.str.strip()

# Sütunları seç
alan = data[['alan']]
fiyat = data[['fiyat']]

# Eğitim ve test setlerini ayır
x_train, x_test, y_train, y_test = train_test_split(alan, fiyat, test_size=0.25, random_state=1)

# Modeli oluştur ve eğit
lr = LinearRegression()
lr.fit(x_train, y_train)

# Tahmin yap
predict = lr.predict(x_test)

# Eğitim verisini sıraya koy
x_train = x_train.sort_index()
y_train = y_train.sort_index()
print(len(x_train))
# Grafik oluştur
plt.figure(figsize=(10, 6))
plt.scatter(x_train, y_train, color="blue", label="Eğitim Verisi")
plt.scatter(x_test, y_test, color="green", label="Test Verisi")
plt.plot(x_test, predict, color="red", label="Tahmin")
plt.xlabel("Alan")
plt.ylabel("Fiyat")
plt.title("Ev Fiyat Tahmini")
plt.legend()
plt.show()
