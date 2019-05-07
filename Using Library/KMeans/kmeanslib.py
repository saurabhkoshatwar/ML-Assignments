import pandas as pd 
import numpy as np 
from sklearn.cluster import KMeans  
import matplotlib.pyplot as plt 

dataset = pd.read_csv("kmean.csv")

wcss =[]
for i in range(1,8):
    kmean_t = KMeans(n_clusters=i)
    kmean_t.fit(dataset)
    wcss.append(kmean_t.inertia_)
plt.plot(range(1,8),wcss)
plt.show()    


kmeans = KMeans(n_clusters=2)
kmeans.fit(dataset)
print(kmeans.labels_)

plt.scatter(dataset.iloc[:,0],dataset.iloc[:,1],c=kmeans.labels_)
plt.show()


