import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from scipy.stats import multivariate_normal


# Seperate data by class
def class_sorted_data(dataset):
    classes = np.unique(dataset[:, np.size(dataset, 1) - 1])
    sortedclassdata = []
    for i in range(len(classes)):
        item = classes[i]
        itemindex = np.where(dataset[:, np.size(dataset, 1) - 1] == item)   
        singleclassdataset = dataset[itemindex, 0:np.size(dataset, 1) - 1]  
        sortedclassdata.append(np.matrix(singleclassdataset))               
    return sortedclassdata, classes

                                     
def get_prior(y):    
    y_unique = np.unique(y)
    len_classes = len(y_unique)
    priors = [None] * len_classes
    for i in range(len_classes):
        label = y_unique[i]
        mask = (y == label)
        len_samples_in_class = np.sum(mask)
        prior_of_class = len_samples_in_class / len(y)
        priors[i] = prior_of_class
    return(priors)

def kerndensitymodel(dataset, classes, h, x, prior):
    probs = np.zeros(shape=(len(dataset),1))

    sigma = np.array([[0.36,0],[0,0.36]])

    numClasses = len(dataset)
    D = dataset[0][0].shape[1]
    
    for i in range(numClasses):
        prob = 0
        N = len(dataset[i])
        for j in range(N):
            u =(x-dataset[i][j])/h
            exponent = (-1/2) * (u @ np.linalg.inv(sigma) @ u.T)
            base = 1 / (2 * np.pi) * ((np.linalg.det(sigma)) ** (1 / 2))
            prob = prob + base * np.exp(exponent)
        prob = prob / (N *(h*h))
        probs[i] = prob * prior[i]
        

    classPrediction = classes[np.argmax(probs)]
    return classPrediction

def trainAccuracy(train_set,prior, h):
    numCorrect = 0
    for i in range(len(train_set)):
        TrueValue = train_set[i][-1]
        x = np.delete(train_set, -1, axis=1)[i]
        trainer = np.delete(train_set, i, axis=0)
        sortclassdata, classes = class_sorted_data(trainer)
        if kerndensitymodel(sortclassdata, classes, h, x ,prior) == TrueValue:
            numCorrect = numCorrect + 1
    accuracy = 100 * (numCorrect/ len(train_set))
    return accuracy

def testAccuracy(train_set, test_set,prior, h):
    sortclassdata, classes = class_sorted_data(train_set)
    numCorrect = 0
    for i in range(len(test_set)):
        TrueValue = test_set[i][-1]
        x = np.delete(test_set, -1, axis=1)[i]
        if kerndensitymodel(sortclassdata, classes, h, x,prior) == TrueValue:
            numCorrect = numCorrect + 1
    accuracy = 100 * (numCorrect/ len(test_set))
    return accuracy



#load dataset

N = 500

mu1 = np.array([2,5])
mu2 = np.array([8,1])
mu3 = np.array([5,3])
cov1 = np.array([[2,0],[0,2]])
cov2 = np.array([[3,1],[1,3]])
cov3 = np.array([[2,1],[1,2]])


dist1_2d = np.random.multivariate_normal(mu1, cov1, N)
dist2_2d = np.random.multivariate_normal(mu2, cov2, N)
dist3_2d = np.random.multivariate_normal(mu3, cov3, N)

y1 = np.full((500,1), 1, dtype=int)
y2 = np.full((500,1), 2, dtype=int)
y3 = np.full((500,1), 3, dtype=int)

class1 =  np.concatenate((dist1_2d, y1), axis=1)
class2 =  np.concatenate((dist2_2d, y2), axis=1)
class3 =  np.concatenate((dist3_2d, y3), axis=1)

dataset = np.concatenate([class1, class2, class3], axis=0)

X_train, X_test = train_test_split(dataset,test_size=0.1)

h=0.1

prior = get_prior(dataset[:,-1])

accuracy = trainAccuracy(X_train ,prior, h)
print(f"Training Accuracy {accuracy}% with h = {h}")

accuracy = testAccuracy(X_train, X_test, prior, h)
print(f"Testing Accuracy {accuracy}% with h = {h}")
