import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder , OneHotEncoder
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm

data= pd.read_csv("C:/Users/burak/OneDrive/Masaüstü/ML/csv_files/veriler.csv")
print(data)
print(data.columns)
print(len(data))

ınfo = data.iloc[:,1:4].values


# kategorik değerleri numerik olarak çevirmemizi sağlar. çünkü regresyonda kategorik değerler 
# kullanamayız numerik değerler kullanmak zorundayız.

ulke = data.iloc[:,0:1].values
print(ulke)

le = LabelEncoder()
ulke[:,0] = le.fit_transform(data.iloc[:,0])
print(ulke)

ohe = OneHotEncoder()
ulke = ohe.fit_transform(ulke).toarray()
print(ulke)


cinsiyet = data.iloc[:,-1:].values
print(cinsiyet)

cinsiyet[:,-1] = le.fit_transform(data.iloc[:,-1])
print(cinsiyet)

cinsiyet = ohe.fit_transform(cinsiyet).toarray()
print(cinsiyet)

# dataframe' e ceviriyoruz.

gender = pd.DataFrame(data = cinsiyet[:,0],index=range(22),columns=['cinsiyet'])

country = pd.DataFrame(data = ulke, index=range(22), columns=['fr','tr','us'])

ınfo = pd.DataFrame(data = ınfo , index=range(22),  columns=['boy','kilo','yas'])


s1 = pd.concat([country,ınfo] , axis=1)

s2 = pd.concat([s1,gender] , axis=1)

print(s2)

x_train,x_test,y_train,y_test = train_test_split(s1,gender,test_size=0.33,random_state=2)

x_train = x_train.sort_index()
y_train = y_train.sort_index()

lr = LinearRegression()
lr.fit(x_train,y_train)

y_pred = lr.predict(x_test)

# diğer bağımsız değişkenlere göre boy(bağımlı değişken) değerine tahmin yapılması
boy = s2.iloc[:,3:4].values
print(boy)

sol = s2.iloc[:,:3]
sağ = s2.iloc[:,4:]

datavet = pd.concat([sol,sağ],axis=1)

x_train,x_test,y_train,y_test = train_test_split(datavet,boy,test_size=0.33,random_state=2)

lr = LinearRegression()
lr.fit(x_train,y_train)

y_predict = lr.predict(x_test)

# backward elimination - model için istatiksel veri oluşturma

X = np.append(arr = np.ones((22,1)).astype(int), values = datavet, axis=1 )

X_l = datavet.iloc[:,[0,1,2,3,4,5]].values
X_l = np.array(X_l,dtype = float)
model = sm.OLS(boy, X_l).fit()
print(model.summary())
