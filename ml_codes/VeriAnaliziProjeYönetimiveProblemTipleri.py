import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder , OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# veriler
data = pd.read_csv("C:/Users/burak/OneDrive/Masaüstü/ML/csv_files/veriler.csv")
print(data)

boykilo = data[['boy','kilo']]
print(boykilo)

# eksik veriler
data1 = pd.read_csv("C:/Users/burak/OneDrive/Masaüstü/ML/csv_files/eksikveriler.csv")
imputer = SimpleImputer(missing_values=np.nan, strategy="mean")

yas = data1.iloc[:,1:4].values
print(yas)
yas1= data1.iloc[:,1:4].values
imputer = imputer.fit(yas[:,1:4])    # fit fonks. öğrenecek olan değerler,eğitilmek için kullanılır.
yas[:,1:4] = imputer.transform(yas[:,1:4])  # öğrenilmiş olan bilgiyi istenilen yere getirmek.
print(yas)

# kategorik ayırma
ulke = data.iloc[:,0:1].values
print(ulke)

le = LabelEncoder()
ulke[:,0] = le.fit_transform(data.iloc[:,0])
print(ulke)

ohe = OneHotEncoder()
ulke = ohe.fit_transform(ulke).toarray()
print(ulke)

# verileri birleştirme

sonuc = pd.DataFrame(data = ulke, index=range(22), columns=['fr','tr','us'])
print(sonuc)

sonuc1 = pd.DataFrame(data = yas1 , index=range(22),  columns=['boy','kilo','yas'])
print(sonuc1)

cinsiyet = data.iloc[:,-1].values
print(cinsiyet)

sonuc2 = pd.DataFrame(data = cinsiyet, index =range(22), columns=['cinsiyet'])
print(sonuc2)

s = pd.concat([sonuc,sonuc1] , axis=1)
print(s)

table = pd.concat([s,sonuc2], axis=1)
print(table)


# veri kümesinin test ve train olarak bölünmesi

x_train,x_test,y_train,y_test =train_test_split(s,sonuc2,test_size=0.33,random_state=0)

sc = StandardScaler()

X_train = sc.fit_transform(x_train)
X_test = sc.fit_transform(x_test)





