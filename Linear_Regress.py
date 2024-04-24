import numpy as np
import matplotlib.pyplot as mtp
import pandas as pd
data_set = pd.read_csv(r'simplelinearregression.csv')
print(data_set.describe())

df=pd.DataFrame(data_set)
print(df.to_string())
# pick columns for x and y
X= data_set.iloc[:, :-1].values
y = data_set.iloc[:, 1].values

print("X=\n",X)
df2=pd.DataFrame(X)
print("X Data-Age")
print(df2.to_string())
df3=pd.DataFrame(y)
print("Y Data-Premium")
print(df3.to_string())
print("Y array\n")
print(y)

#load dataset slicing module
from sklearn.model_selection import train_test_split
# Splitting the dataset into training and test set .
x_train, x_test, y_train, y_test= train_test_split(X, y, test_size= 0.2, random_state=0)

#load liniear regression class
from sklearn.linear_model import LinearRegression
#create an instance of linear regression
regressor= LinearRegression()

regressor.fit(x_train, y_train)
x_pred= regressor.predict(x_train)
print("Prediction result on Test Data")
y_pred = regressor.predict(x_test)

dfs=pd.DataFrame(x_test)
print("X-test")
print(dfs)

df2 = pd.DataFrame({'Actual Y-Data': y_test, 'Predicted Y-Data': y_pred})

print(df2.to_string())

print("Mean")
print(df['Premium'].mean())
from sklearn import metrics

print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
print("Mean")
print(df['Premium'].mean())
from sklearn import metrics

print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

r2_score = regressor.score(x_test,y_test)
print(r2_score*100,'%')

y_pred2 = regressor.predict([[26]])
print("Age 26 ")
print(y_pred2)
#print(type(x_test))
arr=np.array([[18000],[21000]])
print("arr \n")
print(arr)
y_pred3 = regressor.predict(arr)
print(y_pred3)