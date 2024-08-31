# Non Parametric Density Estimation
For this project, we generate a dataset for three classes each with 500 samples from three Gaussian distribution described below:

$$ class1:\quad\mu = \binom{2}{5} \qquad 
\sum =
\begin{pmatrix}
2 & 0 
\\
0 & 2
\end{pmatrix}
$$

$$ class2:\quad\mu = \binom{8}{1} \qquad 
\sum =
\begin{pmatrix}
3 & 1
\\
1 & 3
\end{pmatrix}
$$

$$ class3:\quad\mu = \binom{5}{3} \qquad 
\sum =
\begin{pmatrix}
2 & 1
\\
1 & 2
\end{pmatrix}
$$

Use generated data and estimate the density without pre-assuming a model for the distribution which is done by a non-parametric estimation.
Implement the Gaussian kernel PDF estimation methods using Standard Deviations of 0.2, 0.6, 0.9. Estimate P(X) and Plot the estimated PDF.

### Gaussian kernel Density 3D
![Gaussian kernel density 3d](https://github.com/Ghafarian-code/Gaussian-kernel-Non-Parametric-Density-Estimation/blob/master/image/figure.png)

Also we find the best value for h in the Gaussian kernel model with the standard deviation of 0.6 using 5-Fold cross-validation and for this goal the squared error between the actual function and the estimated Gaussian kernel function should be minimized.                                                      
Then employ the estimated Gaussian kernel for each class and do the followings with standard
deviation 0.6:                                                                               
a) Divide the samples into a 90% train and 10% test data randomly.                           
b) Use Bayesian estimation and predict the class labels while reporting train, test and
total accuracies.
