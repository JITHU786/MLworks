import numpy as np
import matplotlib.pyplot as mtp
import pandas as pd
from sklearn.metrics import accuracy_score

# importing datasets
data_set=pd.read_csv(r'Social_Network_Ads.csv')
df=pd.DataFrame(data_set)
print(df.to_string())

# extracting independent and Dependent variable
x=data_set.iloc[:,[2,3]].values
y=data_set.iloc[:,4].values

# splitting dataset into training and test set.
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=0)

# feature scaling
from sklearn.preprocessing import StandardScaler
st_x=StandardScaler()
x_train=st_x.fit_transform(x_train)
x_test=st_x.transform(x_test)

print("X-train After transform")
df2=pd.DataFrame(x_train)
print(df2.to_string())
print("X-test After transform")
df3=pd.DataFrame(x_test)
print(df3.to_string())

from sklearn.svm import SVC # "SUPPORT VECTOR CLASSIFIER"
classifer=SVC(kernel='linear',random_state=0)
classifer.fit(x_train,y_train)

# Predicting the test set result
y_pred=classifer.predict(x_test)

df2=pd.DataFrame({"Actual Y_test":y_test,"Prediction Data":y_pred})
print("Prediction status")
print(df2.to_string())

# Evaluate predictions
accuracy=accuracy_score(y_test,y_pred)
print('Accuracy: %.2f' % (accuracy*100))

test=[[49,65000]]
test=st_x.transform(test)
df7=pd.DataFrame(test)
print(df7.to_string())

y_pred_2=classifer.predict(test)
print(y_pred_2)