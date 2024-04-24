import numpy as np
import matplotlib.pyplot as mtp
import pandas as pd
from sklearn.metrics import accuracy_score

# importing datasets

data_set=pd.read_csv(r'Naive-Bayes-Classification-Data.csv')
df=pd.DataFrame(data_set)
print(df.to_string())

#selecting features
x=data_set.iloc[:,[0,1]].values
y=data_set.iloc[:,2].values

# Splitting the dataset into the Training set and Test set
from sklearn. model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.transform(x_test)

# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import GaussianNB
classifier=GaussianNB()
classifier.fit(x_train,y_train)

#Prediction of the test set result:
# Predicting the Test set results
y_pred=classifier.predict(x_test)
print("PREDICTION")
df2=pd.DataFrame({"Actual Result-y":y_test,"Prediction Result":y_pred})
print(df2.to_string())

from sklearn import metrics
print('Mean Absolute Error:',metrics.mean_absolute_error(y_test,y_pred))
print('Mean Squared Error:',metrics.mean_squared_error(y_test,y_pred))
print('Root Mean squared Error:',np.sqrt(metrics.mean_squared_error(y_test,y_pred)))

# evaluate predictions
accuracy=accuracy_score(y_test,y_pred)
print('Accuracy: %2f' %(accuracy*100))