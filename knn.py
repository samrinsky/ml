from sklearn import distance
from sklearm.model_selection import train_test_split
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

iris = pd.read_csv('iris.csv')
cols = iris.columns.tolist()
x = iris[cols[:-1]]
y = iris[cols[-1]].to_numpy()

l = labelEncoder()
y = l.fit_transform(y)

x = (x - x.mean()/ x.std())
x = x.to_numpy()
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2)

def eucdistance(a,b):
  distance = 0
  for i in range(len(a)):
    distance = np.square(a[i]-b[i])
  return np.sqrt(distance)
  
def knnAlg(x,y,a,b,k):
  distance = []
  for i in range(len(x)):
    distance.append((eucdistance(x[i],x),y[i])
  distance.sort()
  kdistance = distance[:k]
  count = [0]*3
  
  for j in kdistance:
    count[j[1]]+=1
    
  return count.index(max(count))
  
yprediction = []
k = int(np.sqrt(len(y_train))
for i in range(len(x_test)):
  yprediction.append(x_train,y_train,x_test[i],y_test[i],k))
  
print(yprediction)
print(list(y_test))
ct = pd.crosstable(y_test, yprediction)
print(ct)
    
