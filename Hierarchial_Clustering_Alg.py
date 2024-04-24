# import libraries
import numpy as nm
import matplotlib.pyplot as mtp
import pandas as pd
# Importing the dataset

dataset = pd.read_csv(r'Credit Card Customer Data.csv')
dataset.dropna(inplace=True)
#Form a dataframe
df=pd.DataFrame(dataset)
print(df.to_string())

#Extracting the matrix of features
x = dataset.iloc[:, [5, 6]].values #Annual income and spending score
#Finding the optimal number of clusters using the Dendrogram
import scipy.cluster.hierarchy as shc
dendro = shc.dendrogram(shc.linkage(x, method="ward"))

mtp.title("Dendrogrma Plot")
mtp.ylabel("Euclidean Distances")
mtp.xlabel("Customers")
mtp.show()

#training the hierarchical model on dataset
from sklearn.cluster import AgglomerativeClustering
hc= AgglomerativeClustering(n_clusters=5, affinity='euclidean', linkage='ward')
y_pred= hc.fit_predict(x)
print(y_pred)