
import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D




def my_kde_bivariate(data_2d, h ,sigma):

    x = np.linspace(np.amin(data_2d[:,0]), np.amax(data_2d[:,0]), 50).reshape(-1,1)
    y = np.linspace(np.amin(data_2d[:,1]), np.amax(data_2d[:,1]), 50).reshape(-1,1)
    
    xx, yy = np.meshgrid(x,y)
    X_2d = np.concatenate([xx.ravel().reshape(-1,1), yy.ravel().reshape(-1,1)], axis=1)

    
    N = np.size(X_2d,0)
    d = np.size(data_2d,1) # dimensions of original data set
    probs = [] # store kernel densities for each location
    
    
    # get density estimates for each row in X
    for x in X_2d:
        px = kerndensitymodel(data_2d,h,x,sigma) 
        probs.append(px) 
    
    return xx, yy, probs

def kerndensitymodel(dataset, h, x, sigma):

    prob = 0
    n = len(dataset)
    for j in range(n):
        u =(x-dataset[j])/h
        exponent = (-1/2) * (u @ np.linalg.inv(sigma) @ u.T)
        base = 1 / (2 * np.pi) * ((np.linalg.det(sigma)) ** (1 / 2))
        prob = prob + base * np.exp(exponent)
    prob = prob / (n *(h**2))
    return prob




mu1 = np.array([2,5])
mu2 = np.array([8,1])
mu3 = np.array([5,3])
cov1 = np.array([[2,0],[0,2]])
cov2 = np.array([[3,1],[1,3]])
cov3 = np.array([[2,1],[1,2]])

N = 500

dist1_2d = np.random.multivariate_normal(mu1, cov1, N)
dist2_2d = np.random.multivariate_normal(mu2, cov2, N)
dist3_2d = np.random.multivariate_normal(mu3, cov3, N)

data_2d = np.concatenate([dist1_2d, dist2_2d,dist3_2d], axis=0)

h=0.5
h_set=[0.09,0.3,0.6]

sd = [None] * 3
sd[0] = np.array([[0.04,0],[0,0.04]])
sd[1] = np.array([[0.36,0],[0,0.36]])
sd[2] = np.array([[0.81,0],[0,0.81]])

fig2d = plt.figure()
fig3d = plt.figure()



i = 1
for h in h_set:
    pos = i
    i +=1
    for s in sd:
        title = "sd = %0.2f, h = %0.2f" % (np.sqrt(s[0,0]),h)
        print(title)
        xx2, yy2, z2 = my_kde_bivariate(data_2d, h,s)
        zz2 = np.array(z2).reshape(xx2.shape)

        ax2d = fig2d.add_subplot(3, 3, pos)
        ax3d = fig3d.add_subplot(3, 3, pos,projection='3d')
        pos = pos+3

        ax2d.contour(xx2,yy2,zz2)
        ax2d.set_title(title)

        ax3d.plot_surface(xx2,yy2,zz2)
        ax3d.set_title(title)


plt.show()



