import pandas as pd 
import numpy as np 
import mataplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from warnings import simplefiter

print("---Marvellous Infosystem by Adesh Vavhal----")

print("---Daibates predictor using Decision tree---")

diabetes=pd.read_csv('diabetes.csv')

print("Columns of Dataset")
print(diabetes.columns)

print("first 5 records of dataset")
print(diabetes.head)

print("Dimension of diabetes data: {}".format(diabetes.shape))

X_train,X_test,Y_train,Y_test = train_test_split(diabetes.loc[:,diabetes.columns !='Outcome'], diabetes['Outcome'],stratiy = diabetes['Outcome'],random_state=66)

logreg =LogisticRegression()

logreg.fit(X_test,Y_train)

print("Accuracy on taning set:{:.3f}".format(logreg.score(X_train,Y_train)))

print("Accuracy on test set:{:.3f}".format(logreg.score(X_test,Y_test)))

logreg =LogisticRegression(C=0.01)

logreg.fit(X_test,Y_train)

print("Accuracy on taning set:{:.3f}".format(logreg.score(X_train,Y_train)))

print("Accuracy on test set:{:.3f}".format(logreg.score(X_test,Y_test)))

print("Feature importances:\n{}".format(.feature_importances_))






























