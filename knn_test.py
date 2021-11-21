#Importing Libraries
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split

#Load iris dataset
iris=datasets.load_iris()
X,y=iris.data,iris.target

#Loading data into train and test set
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=1234)
#print(X_train.shape)
#print(X_train[0])
#print(y.shape) 
#print(X_train.shape,X_test.shape)

#Function for calculation of Euclidean Distance
def euclidean_distance(x1,x2):
    return np.sqrt(np.sum((x1-x2)**2))

# Class KNN
class KNN: 
    #Function for initializing k
    def __init__(self,k):
        self.k=k

    #Function for distance calculation of test point with all training points 
     #and selection k nearest distances
    def distance_cal(self,x,X_train):
        distances=[None]*len(X_train)
        min_distances=[999]*self.k
        y_targets=[9]*self.k
        index=0
        for i in X_train:
            distances[index]=euclidean_distance(x,i)
            if max(min_distances)>distances[index]:
                index1=min_distances.index(max(min_distances))
                min_distances[index1]=distances[index]
                y_targets[index1]=y_train[index]
            index=index+1
        return y_targets

    #Estimating the class from nearest distances
    def target_estimate(self,x,X_train):
        y_targets=self.distance_cal(x,X_train)
        dict_counts={}
        for i in y_targets:
            counting=y_targets.count(i)
            dict_counts[i]=counting
        estimate=max(dict_counts,key=dict_counts.get)
        return estimate

    #Estimate of class for east test point in Test Data set
    def each_test_estimates(self,X_test,X_train):
        estimates=[]
        for i in X_test:
            estimate=self.target_estimate(i,X_train)
            estimates.append(estimate)
        return estimates

    #Calculation of accuracy
    def accuracy(self,X_test,X_train):
        estimates=self.each_test_estimates(X_test,X_train)
        correct=0
        for i in range(len(estimates)):
            if estimates[i]==y_test[i]:
                correct=correct+1
        acc=(correct*100)/(len(estimates))
        return acc

#Choosing best k value by cross validation
p1=KNN(3)
print("k:",p1.k,"accuracy:",p1.accuracy(X_test,X_train))
p2=KNN(5)
print("k:",p2.k,"accuracy:",p2.accuracy(X_test,X_train))
p3=KNN(4)
print("k:",p3.k,"accuracy:",p3.accuracy(X_test,X_train))