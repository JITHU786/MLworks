# Kmeans clustering
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv(r'Credit Card Customer Data.csv')
df = pd.DataFrame(dataset)
print(df.to_string())

#Extracting Independent Variables
x = dataset.iloc[:,[5,6]].values

#finding optimal number of clusters using the elbow method
from sklearn.cluster import KMeans
wcss_list = []

#Using for loop for iterations from 1 to 10.

for i in range(1,11):
    kmeans = KMeans(n_clusters=i,init='k-means++',random_state=42)
    kmeans.fit(x)
    wcss_list.append(kmeans.inertia_)
print(wcss_list)
plt.plot(range(1,11),wcss_list)
plt.title('The Elbow Method Graph')
plt.xlabel('Number of Clusters(k)')
plt.ylabel('WCSS_list')
plt.show()
# training the K-means model on a dataset
kmeans = KMeans(n_clusters=5,init='k-means++',random_state=0)
y_predict = kmeans.fit_predict(x)
print(y_predict)