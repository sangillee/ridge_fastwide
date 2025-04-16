# ridge_fastwide

Updated Apr 16th 2025

This function implements MATLAB's ridge regression with an additional feature of speeding up computations when the number of features outweigh the number of observations greatly. Usage is the same as MATLAB's default ridge function. The function checks the dimension of X to automatically decide whether to use MATLAB's algorithm, which is adding pseudo observations, or to use the new algorithm.

The new algorithm is based on the equality that
(X'*X + k*I)\X' = X'/(X*X' + k*I)
which I learned from the website below
https://danieltakeshi.github.io/2016/08/05/a-useful-matrix-inverse-equality-for-ridge-regression/

For example, if X is a 50-by-1000 matrix
X = rand(50,1000); k = 0.1;
tic; out1 = (X'*X + k*eye(1000))\X'; toc
tic; out2 = X'/(X*X' + k*eye(50)); toc
the first one takes 0.016403 seconds for me while the second one takes 0.001312 seconds
this is because X'*X is a very large matrix (1000 by 1000) while X*X' is quite small (50 by 50)

Using this example, let
X = rand(50,1000); Y = X*rand(1000,1);
tic; b1 = ridge(Y,X,logspace(-3,0,10)); toc  % took 0.894749 seconds
tic; b2 = ridge_fastwide(Y,X,logspace(-3,0,10)); toc  % took 0.010885 seconds
norm(b1-b2)   % less than 1e-07, hence no difference in coefficients

In normal situations when number of observations are greater, they use the same code, hence b should be exactly the same