# -*- coding: utf-8 -*-
"""
Created on Sat Jan 18 22:29:19 2020

#@author Akshat Chauhan
"""



######################################################################
#The codes are based on Python3 
######################################################################
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split

df=pd.read_csv("q3.csv",header=None)

X=df.values[:,0:54]

y=df.values[:,54:55].ravel()

### Ans 3.a.i

# Splitting ino 80% train and 20 % test data 

# please add or remove random_state in below procedure

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

### Naive Bayes Classifier

clf = GaussianNB()
clf.fit(X_train, y_train)
NBy_predicted = clf.predict(X_test)

NB_r=classification_report(y_test,NBy_predicted)

print(NB_r)


### Logistic Regression

clf1 = LogisticRegression(penalty='none').fit(X_train, y_train)
LRy_predicted=clf1.predict(X_test)
LR_r=classification_report(y_test,LRy_predicted)

print(LR_r)


### using nearest neighbor classifier 


clf2 = KNeighborsClassifier(n_neighbors=3)
clf2.fit(X_train, y_train)
KNNy_predicted=clf2.predict(X_test)
KNN_r=classification_report(y_test,KNNy_predicted)

print(KNN_r)

#accuracy_score(y_test,KNNy_predicted)

print("Naive Bayes Classifier gives accuracy .....        {}".format(accuracy_score(y_test,NBy_predicted)))
print("Logistic Regression Classifier gives accuracy .....{}".format(accuracy_score(y_test,LRy_predicted)))
print("Nearest Neighbor Classifier gives accuracy .....   {}".format(accuracy_score(y_test,KNNy_predicted)))

print(accuracy_score(y_test,NBy_predicted),accuracy_score(y_test,LRy_predicted),accuracy_score(y_test,KNNy_predicted))


                            
### Ans 3.a.ii
    
















cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])                   
                            
h = .02
x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),np.arange(y_min, y_max, h))    





## Naive Bayes
pclf = GaussianNB()
pclf.fit(X_train[:,0:2], y_train)   
Z = pclf.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.figure(figsize=(15,10))

plt.pcolormesh(xx, yy, Z, cmap=cmap_light)
plt.scatter(X_train[:, 0], X_train[:, 1],s=25,c=y_train,marker="o",label='Training Data', cmap=cmap_bold)
plt.scatter(X_test[:, 0], X_test[:, 1], s=100,c=y_test,marker="+",label='Test Data',cmap=cmap_bold)
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.title("Naive Bayes classification with Gaussian function")
plt.legend()
plt.show()




## Logistic regression


Pclf1 = LogisticRegression(penalty='none')
Pclf1.fit(X_train[:,0:2], y_train)   
Z = Pclf1.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.figure(figsize=(15,10))

plt.pcolormesh(xx, yy, Z, cmap=cmap_light)
plt.scatter(X_train[:, 0], X_train[:, 1],s=25,c=y_train,marker="o",label='Training Data', cmap=cmap_bold)
plt.scatter(X_test[:, 0], X_test[:, 1], s=100,c=y_test,marker="+",label='Test Data',cmap=cmap_bold)
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.title("Logistic Regression classification with no regularization")
plt.legend()
plt.show()




## Nearest neighbor
Pclf2 = KNeighborsClassifier(n_neighbors=3)
Pclf2.fit(X_train[:,0:2], y_train)   
Z = Pclf2.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.figure(figsize=(15,10))

plt.pcolormesh(xx, yy, Z, cmap=cmap_light)
plt.scatter(X_train[:, 0], X_train[:, 1],s=25,c=y_train,marker="o",label='Training Data', cmap=cmap_bold)
plt.scatter(X_test[:, 0], X_test[:, 1], s=100,c=y_test,marker="+",label='Test Data',cmap=cmap_bold)
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.title("K nearest Neighbor classification with k = 3")
plt.legend()
plt.show()


























###############Q 3.b

import scipy.io as sio

matFile1 = sio.loadmat('data.mat')
data = matFile1['data']


data=data.T
data.shape
data

label_matFile1 = sio.loadmat('label.mat')
label_data = label_matFile1['trueLabel']
label_data=label_data.T

label_data.ravel().shape

lst=[0 if i==2 else 1 for i in label_data]
y=np.array(lst)
#y=y.reshape(len(y),1)
label_data
X=data


X.shape
y.shape


# Splitting ino 80% train and 20 % test data 
#,random_state=0
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=0)

### Naive Bayes Classifier

clf = GaussianNB()
clf.fit(X_train, y_train)
NBy_predicted = clf.predict(X_test)

NB_r=classification_report(y_test,NBy_predicted)

print(NB_r)


### Logistic Regression

clf1 = LogisticRegression(penalty='none').fit(X_train, y_train)
LRy_predicted=clf1.predict(X_test)
LR_r=classification_report(y_test,LRy_predicted)

print(LR_r)


### using nearest neighbor classifier 


clf2 = KNeighborsClassifier(n_neighbors=3)
clf2.fit(X_train, y_train)
KNNy_predicted=clf2.predict(X_test)
KNN_r=classification_report(y_test,KNNy_predicted)

print(KNN_r)

#accuracy_score(y_test,KNNy_predicted)

print("For MNIST data Naive Bayes Classifier gives accuracy .....        {}".format(accuracy_score(y_test,NBy_predicted)))
print("For MNIST data Logistic Regression Classifier gives accuracy .....{}".format(accuracy_score(y_test,LRy_predicted)))
print("For MNIST data Nearest Neighbor Classifier gives accuracy .....   {}".format(accuracy_score(y_test,KNNy_predicted)))

print(accuracy_score(y_test,NBy_predicted),accuracy_score(y_test,LRy_predicted),accuracy_score(y_test,KNNy_predicted))


