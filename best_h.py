import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold



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

x = np.concatenate([dist1_2d , dist2_2d , dist3_2d ], axis=0)
y = np.concatenate([y1 , y2, y3 ], axis=0)

sigma = np.array([[0.36,0],[0,0.36]])

def pdf_multivariate_gauss(x, mu, cov):

    part1 = 1 / (2* np.pi) * (np.linalg.det(cov)**(1/2)) 
    part2 = (-1/2) * ((x-mu).T @ (np.linalg.inv(cov))) @((x-mu))
    return float(part1 * np.exp(part2))


def kerndensitymodel(dataset, h, x):

    sigma = np.array([[0.36,0],[0,0.36]])    
    prob = 0
    n = len(dataset)
    for j in range(n):
        u =(x-dataset[j])/h
        exponent = (-1/2) * (u @ np.linalg.inv(sigma) @ u.T)
        base = 1 / (2 * np.pi) * ((np.linalg.det(sigma)) ** (1 / 2))
        prob = prob + base * np.exp(exponent)
    prob = prob / (n *(h**2))
    return prob


kf = KFold(n_splits=5) 
kf.get_n_splits(x)


h_set = np.linspace(0.1,1,5)
min_error = []
h_best = []
k=1

for train_index, test_index in kf.split(x):
    X_train, X_test = x[train_index], x[test_index] 
    y_train, y_test = y[train_index], y[test_index]

    errors = []
    print("fold",k)
    k = k+1

    for i in range(len(h_set)):
        error = 0
        for j in range(len(X_test)):

            p_j = kerndensitymodel(X_train, h_set[i] , X_test[j])
            
            if(y_test[j]==1):
                p = pdf_multivariate_gauss(X_test[j], mu1, cov1)

            if(y_test[j]==2):
                p = pdf_multivariate_gauss(X_test[j], mu2, cov2)

            if(y_test[j]==3):
                p = pdf_multivariate_gauss(X_test[j], mu3, cov3)
            error = error + ((p_j - p)**2)
            
        errors.append(error/len(X_test))
        #print(errors[i])
    index = np.argmin(errors)
    min_error.append(errors[index])
    h_best.append(h_set[index])

print("best h in each fold:",h_best)
print("nin squared error in each fold:", min_error)
print("best bandwith = ",h_best[np.argmin(min_error)])