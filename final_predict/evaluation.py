import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler

df = pickle.load(open("finaldataset", "rb"))
x = np.arange(0,1924)
df.set_index(x, inplace=True)
df2=pd.DataFrame()
row_prev=0
dataframes=[]
for row in range(len(df['class labels'])-1):
    if df['class labels'][row] != df['class labels'][row+1]:
        df2=df.loc[row_prev+1:row, :]
        row_prev=row
        dataframes.append(df2)

z = []
for frame in dataframes:
	if int(frame.iloc[0,[6]].values) == 116:
		z.append(frame)

framz = []
for i in range(len(z)):
	framz.append(z[i])

frame = pd.concat(framz)

X = frame.iloc[:,[0,1,2,3,4,5]].values
y = frame.iloc[:,[7]].values

scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X)
# we must apply the scaling to the test set that we computed for the training set

knn = KNeighborsClassifier(n_neighbors = 5)
knn.fit(X, y)
print('Accuracy of K-NN classifier on training set: {:.2f}'
     .format(knn.score(X, y)))

