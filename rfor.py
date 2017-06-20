import numpy as np
import pandas as pd 
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import Imputer
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import sys
import time

t=time.time()

#read data from csv
readCSV=pd.read_csv('train.csv')

#store training data into corresponding lists
l=[]
for i in range(0,784):
	l.append('pixel'+str(i))
#print l

#independent variables
X_train=readCSV[l]

#prediction variable
Y_train=readCSV['label']

#to fix Scikit NaN or infinity error message
X_train = Imputer().fit_transform(X_train)

#split dataset into training and testing data
#X_train,X_test,Y_train,Y_test=train_test_split(X_train,Y_train,test_size=0.3)

#Random Forests classifier
rf=RandomForestClassifier(n_estimators=100)
rf.fit(X_train,Y_train)

#load test data
testCSV=pd.read_csv('test.csv')

#store testing data into corresponding lists
l=[]
for i in range(0,784):
	l.append('pixel'+str(i))
#print l

#test data
X_test=testCSV[l]

#classification is done here
prediction=rf.predict(X_test)

print "Prediction of class of test data"
print prediction

l=[]
for i in range(1,28001):
	l.append(str(i))

df1=pd.DataFrame(l, columns=['ImageId'])
df1.to_csv('submission1.csv', index = False)

for i in range(len(prediction)):
	prediction[i]=str(prediction[i])	


df2=pd.DataFrame(prediction, columns=["Label"])
df2.to_csv('submission.csv', index=False)

'''
print "Confusion matrix"
print confusion_matrix(Y_test,prediction)

print "Accuracy value"
print accuracy_score(Y_test,prediction)
'''
print "Executed in "+str(time.time()-t)+"s"
