#Gerekli kütüphanelerin import edilmesi
import pandas as pd
import numpy as np
from sklearn import datasets, svm, metrics


#Digits verisetinin çekilmesi
df=datasets.load_digits()
digits=datasets.load_digits()


n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1))



#Bağımlı ve bağımsız değişken olarak ayrılması
X=data
y=df.target


#Verisetinin train- test olarak ayrılması.
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


#SVM Modeli
model= svm.SVC(kernel='poly',degree=3,gamma='auto_deprecated').fit(X_train,y_train)


y_pred=model.predict(X_test)


#R2 skoru ve Hassasiyet skoru
print(f'R2 score: {r2_score(y_test,y_pred)}')
print(f'Accuracy score: {accuracy_score(y_test,y_pred)}')