import numpy as np
import matplotlib.pyplot as mtp
import pandas as pd
from sklearn.metrics import accuracy_score

# importing datasets
data_set=pd.read_csv(r'Social_Network_Ads.csv')
df=pd.DataFrame(data_set)
print(df.to_string())

#Extracting Independent and dependent Variable
x=data_set.iloc[:,[2,3]].values
y=data_set.iloc[:,4].values

# Splitting the dataset into training and test set.
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=0)

#feature Scaling
from sklearn.preprocessing import StandardScaler
st_x=StandardScaler()

x_train=st_x.fit_transform(x_train)
x_test=st_x.transform(x_test)

#Fitting Decision Tree classifier to the training set
from sklearn.tree import DecisionTreeClassifier
classifier= DecisionTreeClassifier(criterion='entropy', random_state=0)
classifier.fit(x_train, y_train)
#Predicting the test set result
y_pred= classifier.predict(x_test)
df2=pd.DataFrame({"Actual Y_Test":y_test,"Prediction Data":y_pred})
print("Prediction Result")
print(df2.to_string())
from sklearn import metrics
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
from sklearn.metrics import accuracy_score
# evaluate predictions
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy: %.2f' % (accuracy*100))

c_test=[[35,72000]]
k_test=st_x.transform(c_test)
df3=pd.DataFrame(c_test)
print(df3.to_string())

y_pred_2=classifier.predict(c_test)
print(y_pred_2)