from random import random
from sklearn import datasets
from sklearn.model_selection import train_test_split
import numpy as np
from RandomForest import RandomForest
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler


dataset = pd.read_csv('weatherAUS.csv')
X = dataset.iloc[:,[1,2,3,4,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21]].values
Y = dataset.iloc[:,-1].values


Y = Y.reshape(-1,1)

imputer = SimpleImputer(missing_values=np.nan,strategy='most_frequent')
X = imputer.fit_transform(X)
Y = imputer.fit_transform(Y)
# data = datasets.load_breast_cancer()
# X = data.data
# y = data.target
from sklearn.preprocessing import LabelEncoder
le1 = LabelEncoder()
X[:,0] = le1.fit_transform(X[:,0])
le2 = LabelEncoder()
X[:,4] = le2.fit_transform(X[:,4])
le3 = LabelEncoder()
X[:,6] = le3.fit_transform(X[:,6])
le4 = LabelEncoder()
X[:,7] = le4.fit_transform(X[:,7])
le5 = LabelEncoder()
X[:,-1] = le5.fit_transform(X[:,-1])
le6 = LabelEncoder()
Y[:,-1] = le6.fit_transform(Y[:,-1])



Y = np.array(Y)
# print(Y)
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)

Y = Y.astype('int64').flatten()

# print(X)
# print(Y)



X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=1000
)

def accuracy(Y_true, Y_pred):
    accuracy = np.sum(Y_true == Y_pred) / len(Y_true)
    return accuracy

clf = RandomForest(n_trees=20)
clf.fit(X_train, Y_train)
predictions = clf.predict(X_test)

acc =  accuracy(Y_test, predictions)
print(acc)