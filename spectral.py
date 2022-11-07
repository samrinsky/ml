
from sklearn import datasets
import pandas as pd
from sklearn.cluster import SpectralClustering
data = datasets.load_breast_cancer().data

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
data = sc.fit_transform(data)
model = SpectralClustering(n_clusters=2, affinity="rbf")
labels = model.fit_predict(data)
