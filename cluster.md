# cluster
unsupervised--no lable

birch: much quicker

DBSCAN: set the r and p?(nodes number of the circle)--main choice

K-means: set k (how many cluster you want)

## k-means
```
import pandas as pd
beer = pd.read_csv('data.txt', sep=' ')
beer
X = beer[["calories","sodium","alcohol","cost"]]
```
choose k number
```
from sklearn.cluster import KMeans

km = KMeans(n_clusters=3).fit(X)
km2 = KMeans(n_clusters=2).fit(X)
```
each data belongs to which cluster
```
km.labels_
```
```
beer['cluster'] = km.labels_
beer['cluster2'] = km2.labels_
beer.sort_values('cluster')
```
```
from pandas.tools.plotting import scatter_matrix
%matplotlib inline

cluster_centers = km.cluster_centers_

cluster_centers_2 = km2.cluster_centers_
```
get some information about each cluseter you want 
```
beer.groupby("cluster").mean()
beer.groupby("cluster2").mean()
```
set centers and observe it
```
centers = beer.groupby("cluster").mean().reset_index()

%matplotlib inline
import matplotlib.pyplot as plt
plt.rcParams['font.size'] = 14

import numpy as np
colors = np.array(['red', 'green', 'blue', 'yellow'])

plt.scatter(beer["calories"], beer["alcohol"],c=colors[beer["cluster"]])

plt.scatter(centers.calories, centers.alcohol, linewidths=3, marker='+', s=300, c='black')

plt.xlabel("Calories")
plt.ylabel("Alcohol")

```
find the information of features of clusters
```
scatter_matrix(beer[["calories","sodium","alcohol","cost"]],s=100, alpha=1, c=colors[beer["cluster"]], figsize=(10,10))
plt.suptitle("With 3 centroids initialized")
```
```
scatter_matrix(beer[["calories","sodium","alcohol","cost"]],s=100, alpha=1, c=colors[beer["cluster2"]], figsize=(10,10))
plt.suptitle("With 2 centroids initialized")
```
## standarize our data before do cluster
which is important
```
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled
```
```
km = KMeans(n_clusters=3).fit(X_scaled)
beer["scaled_cluster"] = km.labels_
beer.sort_values("scaled_cluster")
beer.groupby("scaled_cluster").mean()
pd.scatter_matrix(X, c=colors[beer.scaled_cluster], alpha=1, figsize=(10,10), s=100)
```

## evaluation
Silhouette Coefficient

calculate a node (ai)--the distant of this node to other nodes within cluster

calculate a node (bi)--the distant of this node to other nodes out of this cluster

S(i) close to 1, much better  (understand it as correlation coefficient)

```
from sklearn import metrics
score_scaled = metrics.silhouette_score(X,beer.scaled_cluster)
score = metrics.silhouette_score(X,beer.cluster)
print(score_scaled, score)
```
try to find the the best k value
```
scores = []
for k in range(2,20):
    labels = KMeans(n_clusters=k).fit(X).labels_
    score = metrics.silhouette_score(X, labels)
    scores.append(score)

scores
```
```
plt.plot(list(range(2,20)), scores)
plt.xlabel("Number of Clusters Initialized")
plt.ylabel("Sihouette Score")
```

## DBSCAN

better used in irregulat data

eps=r min_sample=least q
```
from sklearn.cluster import DBSCAN
db = DBSCAN(eps=10, min_samples=2).fit(X)
```
same us we did in k-means etc..
```
labels = db.labels_
beer['cluster_db'] = labels
beer.sort_values('cluster_db')
beer.groupby('cluster_db').mean()
pd.scatter_matrix(X, c=colors[beer.cluster_db], figsize=(10,10), s=100)
```
