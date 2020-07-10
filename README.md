# Regression and Classification with Gradient Descent

This repository contains the python implementation of two machine learning models (Linear regression, Logistic regression) using gradient descent (optimization algorithm used in machine learning).

And the data is form Machine learning stanford course on coursera (by Andrew ng).

# Gradient Descent mathematically

	θj := θj − α(1/m) * sum[(hθ(x(i)) − y(i)) * (x(i)(j)))]

The same equation is used for both models


# Regression model

# Linear regression

The hypothesis function for linear regression:

	hθ(x) = θ(transpose) * X = θ0 + θ1 * X1 + θ2 * X2 + ..... + θn * Xn

The loss functions for linear regression (mean squared error):

	J(θ) = 1/2*m * sum[hθ((x(i)) − y(i))^2]



# Classification model

# Logistic regression

The hypothesis function for logistic regression:
	
	hθ(x) = g(θ(transpose) * X)
	
	g(z) = 1 / (1 + e^-z)

The loss function for logistic regression:
	
	J(θ) = 1/m * sum[(−y(i) * log(hθ(x(i)))) − ((1 − y(i)) log(1 − hθ(x(i))))]

# Hints

θ: The parameters for the hypothesis function or some people call them the weights.

m: The number of rows in your data.

n: The number of features(columns) in your data.

x(i)(j) : X super i, sub j.


## Requirements 

These models have been tested and developed using the following libraries: 

    - python==3.8.2
    - numpy==1.13.3
    - scipy==1.0.0
    - matplotlib==2.1.2
    
I recommend using at least these versions of the required libraries or later. Python 2 is not supported. 
 

## Caveats and tips

-  In `numpy`, the function `dot` is used to perform matrix multiplication. The operation '*' only does element-by-element multiplication. If you are using python version 3.5+, the operator '@' is the new matrix multiplication, and it is equivalent to the `dot` function.
