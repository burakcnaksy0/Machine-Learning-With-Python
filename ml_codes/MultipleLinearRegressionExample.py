import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder , OneHotEncoder
import statsmodels.api as sm


data= pd.read_csv("C:/Users/burak/OneDrive/Masaüstü/ML/csv_files/odev_tenis.csv")

print(data.columns)
print(data)

# We change the categorical data to convert it into numerical values.

outlook = data[['outlook']].values
windy = data[['windy']].values
play = data[['play']].values

le = LabelEncoder()
ohe = OneHotEncoder()
    
outlook[:,0] = le.fit_transform(data.iloc[:,0])
#print(outlook) 
outlook = ohe.fit_transform(outlook).toarray()  
#print(outlook) 

windy[:,0] = le.fit_transform(data.iloc[:, 3:4])
#print(windy)
windy = ohe.fit_transform(windy).toarray()
#print(windy)

play[:,0] = le.fit_transform(data.iloc[:,-1])
#print(play)
play=  ohe.fit_transform(play).toarray()
#print(play)


# Now let's convert it to dataframes.

outlook = pd.DataFrame(data = outlook , index = range(len(data)) , columns=['overcast','rainy','sunny'])
windy = pd.DataFrame(data = windy[:,0] , index=range(len(data)) , columns=['windy'])  # true(0) or false(1)
play= pd.DataFrame(data = play[:,0] , index=range(len(data)) , columns=['play'])  # yes(0) or no(1)


windy_play = pd.concat([windy,play], axis=1)
temp_hum  = data.iloc[:,1:3]  
notOutlook = pd.concat([temp_hum,windy_play], axis=1)
totalData = pd.concat([outlook,notOutlook],axis=1)


# Let's create our test and train sets
print(totalData)

sol = totalData.iloc[:,0:3]    # 0 included not 3
sağ = totalData.iloc[:,4:6]     # 4 inclusive and after that

temp = totalData[['temperature']]   # bağımlı değişkenimiz olarak bunu seçtik

independentVariables = pd.concat([sol,sağ],axis=1)

x_train , x_test , y_train , y_test = train_test_split(independentVariables, temp, test_size=0.33,random_state=2)

lr = LinearRegression()
lr.fit(x_train, y_train)

y_pred = lr.predict(x_test)


# By calculating the p-value, we remove the value that worsens the prediction.
X = np.append(arr = np.ones((14,1)).astype(int), values = independentVariables, axis=1 )

X_l = independentVariables.iloc[:,[0,1,2,3,4]].values
X_l = np.array(X_l,dtype = float)
model = sm.OLS(temp, X_l).fit()
print(model.summary())

plt.scatter(y_pred,y_test,color="blue")
plt.plot([min(y_pred), max(y_pred)], [min(y_pred), max(y_pred)], color='red', linewidth=2)
plt.show()

