import pandas as pd 
import numpy as np 
import mataplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

print("---Marvellous Infosystem by Adesh Vavhal----")

print("---Daibates predictor using Random forest---")

diabetes=pd.read_csv('diabetes.csv')

print("Columns of Dataset")
print(diabetes.columns)

print("first 5 records of dataset")
print(diabetes.head)

print("Dimension of diabetes data: {}".format(diabetes.shape))

X_train,X_test,Y_train,Y_test = train_test_split(diabetes.loc[:,diabetes.columns !='Outcome'], diabetes['Outcome'],stratiy = diabetes['Outcome'],random_state=66)

rf=RandomForestClassifier(n_estimators = 100,random_state=0)

rf.fit(X_test,Y_train)

print("Accuracy on taning set:{:.3f}".format(rf.score(X_train,Y_train)))

print("Accuracy on test set:{:.3f}".format(rf.score(X_test,Y_test)))

rf=RandomForestClassifier(max_depth =3,n_estimators = 100,random_state=0)

rf.fit(X_test,Y_train)

print("Accuracy on taning set:{:.3f}".format(rf.score(X_train,Y_train)))

print("Accuracy on test set:{:.3f}".format(rfscore(X_test,Y_test)))

print("Feature importances:\n{}".format(rf.feature_importances_))

def plot_feature_importances_diabetes(model):
    plt.figure(figsize =(8.6))
    n_features = 8
    plt.barh(range(n_features),model.feature_importances_,align='centre')
    diabetes_features=[x for i,x in enumerate(diabetes.columns)if !=8]
    plt.yticks(np.arange(n_features),diabetes_features)
    plt.xlabel("Feature importance")
    plt.ylabel("Feature")
    plt.ylim(-1,n_features)
    plt.show()

plot_feature_importances_diabetes(rf)






























