import pandas as pd 
import numpy as np 
import mataplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn.neighbor import KNeighborsClassifier

print("---Marvellous Infosystem by Adesh Vavhal----")

print("---Daibates predictor using K nearest neighobrs---")

diabetes=pd.read_csv('diabetes.csv')

print("Columns of Dataset")
print(diabetes.columns)

print("first 5 records of dataset")
print(diabetes.head)

print("Dimension of diabetes data: {}".format(diabetes.shape))

X_train,X_test,Y_train,Y_test = train_test_split(diabetes.loc[:,diabetes.columns !='Outcome'], diabetes['Outcome'],stratiy = diabetes['Outcome'],random_state=66)

traning_accuracy =[]
test_accuracy=[]

#try n_Neighbors from 1 to 10
neighbors_setting=range(1,10)

for n_neighbors in neighbor_setting:
    #bulid the model
    knn =KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(X_test,Y_train)
    #record tranning set acuracy
    traning_accuracy.append(knn.score(X_train,Y_train))
    #record test set acuracy
    test_accuracy.append(knn.score(X_test,Y_test))

plt.plot(neighbors_setting,test_accuracy,label=traning_accuracy)
plt.plot(neighbors_setting,test_accuracy,label=test_accuracy)
plt.ylabel=("Accuracy")
plt.xlabel=("n_neighbors")
plt.legend()
plt.savefig('knn_compare_model')
plt.show()

knn=KNeighborsClassifier(n_neighbors=9)
knn.fit(X_train,Y_train)

print("Accuracy of knn classifier on tranning set:{:.2f}".format(knn.score(X_train,Y_train)))

print("Accuracy of knn classifier on test set:{:.2f}".format(knn.score(X_test,Y_test)))





























