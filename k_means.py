import numpy as np
import pandas as pd
import math
from scipy.spatial import distance
from sklearn.metrics import silhouette_score
from sklearn.metrics import davies_bouldin_score as db_score


#kmeans
def kmeans(x,k):
  # randomly initialise centroids
  indices = np.arange(x.shape[0])
  np.random.seed(1)
  choices = np.random.choice(indices,k)
  centroids = np.empty((k, x.shape[1]))
  for i in range(k):
    centroids[i] = x[choices[i]]
  # iterate till termination condition
  i=0
  while (True): # new_centroid - centroid < eps
    if i!=0:
      centroids = new_centroids
    i+=1
    dists = find_dists(x,centroids,k) # x[0] X c
    labels = find_labels( x,dists,k)
    new_centroids = find_center(x, labels,k)
    diff =0
    for j in range(k):
      diff += distance.euclidean(centroids[j],new_centroids[j])
    if i>10: #termination condition (can be no. of iterations or if previous round clusters are same)
      break
  return labels,centroids

def find_dists(x,centers,k):
  dists = np.empty((x.shape[0],k))
  for i in range(x.shape[0]):
    for j in range(k):
      dists[i][j] = distance.euclidean(x[i],centers[j])
  return dists

def find_center(x,labels,k):
  new_centers = []
  for i in range(k):
    num = np.zeros(x.shape[1])
    den = 0
    for j in range(x.shape[0]):
      if labels[j] == i:
        den+=1
        num = np.add(num,x[j])
    if den ==0:
      cen = num
    else:
      cen = num/den
    new_centers.append(cen)
  return np.array(new_centers)


def find_labels(x,dists,k):
  labels = np.empty((x.shape[0],))
  for i in range(x.shape[0]):
    index = np.where(dists[i] == np.amin(dists[i]))
    labels[i] = index[0][0]
  return labels

#driver code to run for 56 datasets
sil = []
db = []
for j in range(1,57):
  df = pd.read_csv(str(j)+".csv",header=None)
  X = df.to_numpy()
  X = X[:,:-1]
  # print(np.isnan(X).any() and np.isinf(X).any())
  sil_each = []
  db_each = []
  np.random.seed(j)
  for i in range(2,11):
    cmeans = kmeans(X,i)
    # print("done")
    sil_each.append(silhouette_score(X,cmeans[0]))
    db_each.append(db_score(X,cmeans[0]))
  sil.append(sil_each)
  db.append(db_each)

# save to csv
sil = np.array(sil)
db = np.array(db)
sil_db = np.concatenate((sil,db),axis=1)
print(sil_db.shape)
DF = pd.DataFrame(sil_db)
DF.to_csv("sil_db_kmeans.csv")