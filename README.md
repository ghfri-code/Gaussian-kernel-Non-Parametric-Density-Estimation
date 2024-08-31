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
Implement the Gaussian kernel PDF estimation methods using Standard Deviations of 0.2,0.6,0.9. Estimate P(X) and Plot the true and estimated PDF.
### True Density 3D
![true density 3d](https://github.com/Ghafarian-code/Parzen-Window-Non-Parametric-Density-Estimation/blob/main/images/Figure_2.png)
### Gaussian kernel Density 3D
![Gaussian kernel density 3d](https://github.com/Ghafarian-code/Parzen-Window-Non-Parametric-Density-Estimation/blob/main/images/Figure_4.png)

• Estimate P(X) and Plot the true and estimated PDF and compare them for each of the
separate model.
• Find the best value for h in the Gaussian kernel model with the standard deviation of 0.6
using 5-Fold cross-validation.
• Plot the bias and variance functions of estimated PDFs of class2, using Gaussian kernel
with standard deviation of 0.2 and 0.6 and KNN estimator with k=1 and k=99.
• Employ the estimated Gaussian kernel for each class and do the followings with standard
deviation 0.6:
a) Divide the samples into a 90% train and 10% test data randomly.
b) Use Bayesian estimation and predict the class labels while reporting train, test and
total accuracies.