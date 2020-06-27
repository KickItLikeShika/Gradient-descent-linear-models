import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize


def gradientDescent(X, y, theta, alpha, iterations):
    """Making optimization using gradient descent."""
    m = y.shape[0]

    # Saving all the loss in a list to see how was our progress
    jh = []

    n = theta.shape[0]

    for i in range(iterations):
        # Copy the thetas to make our changes
        # on thetas simealtiouesly
        temp = theta.copy()

        for j in range(n):
            # Compute our predicted values with our last theta values
            hypothesis = sigmoid(np.dot(X, theta.T))
            # Subtract our predicted values from the actual
            # Values to get the error
            error = np.subtract(hypothesis, y)
            # Multiply the errors with its X values
            term = error @ X[:, j]
            # Sum all the term
            summation = np.sum(term)
            # get the gradient
            gradient = (alpha * summation) / m
            # Get the new theta
            temp[j] -= gradient
        
        # Save our last theta values
        theta = temp
        j = lossFunction(theta, X, y)
        # Print our loss to see our progress
        # print("Cost Funciton: ", j)
        # Savin the loss function values in our list
        jh.append(j)

    return theta, jh


def gradient_descent(theta, X, y, m, n, alpha, iterations):
    hypothesis = sigmoid(np.dot(X, theta.T))
    summation = 0
    error = np.subtract(hypothesis, y)
    term = error @ X
    # summation = np.sum(term)
    d = np.divide(term, m)
    # c = np.multiply(d, alpha)
    # gradient = (alpha * summation) / m
    theta = d

    return theta



def lossFunction(theta, X, y):
    """Computing the cost and how good our model was."""
    m = y.size
    J = 0
    ones = np.ones(m)
    # Compute our hypothesis
    fhypothesis = sigmoid(np.dot(theta, X.T))
    # Get the right term
    right_term = np.log(np.subtract(ones, fhypothesis))
    fright_term = (np.subtract(ones, y) @ right_term)
    # Get the left term
    fleft_temp = -y @ np.log(fhypothesis)
    summation = np.sum((np.subtract(fleft_temp, fright_term)))
    # Get the final cost
    j = (1/m) * summation
    return j


def sigmoid(z):
    """Coumpute the hypothesis."""
    return 1/(1 + np.exp(-z))


def plot_data(data):
    """Plot all the data."""
    X, y = data[:, :2], data[:, 2]
    # Classify the the data with y = 1 and y = 0
    pos = y == 1
    neg = y == 0
    plt.plot(X[pos, 0], X[pos, 1], 'k+', 'linewidth', 2, c='red')
    plt.plot(X[neg, 0], X[neg, 1], 'ko', 'MarkerFaceColor', c='blue')
    plt.show()


def main():
    # Load the data
    data = np.loadtxt('Data/ex2data1.txt', delimiter=',')
    # Split the data and get our features and our target
    X, y = data[:, :2], data[:, 2]
    plot_data(data)
    # Test sigmoid
    z = np.array([0, 1, -10000000, 10000000000])
    s = sigmoid(z)
    print("Test the sigmoid function: ", sigmoid)
    # The number of rows and columns
    m, n = X.shape
    # Insert the X0 column
    X = np.concatenate([np.ones((m, 1)), X], axis=1)
    # Initialize the thetas with zeros
    init_theta = np.zeros(n+1)
    # Get the loss before the optimization
    J = lossFunction(init_theta, X, y)
    print("The cost before making optimization: ", J)
    # The learning rate
    alpha = 0.01
    # The number of iterations
    iterations = 1000
    # The test theta
    test_theta = np.array([-24, 0.2, 0.2])
    # Gradient with the zero thetas
    the = gradient_descent(init_theta, X, y, m, n, alpha, iterations)
    # the, j = gradientDescent(X, y, init_theta, alpha, iterations)
    print("Gradient theta at zero: ", the)
    # Gradient with the test theta
    the = gradient_descent(test_theta, X, y, m, n, alpha, iterations)
    print("Gradient theta at test_theta: ", the)
    # We can optimize the parameters using scipy built in function
    """
    Optimize using scipy
    set options for optimize.minimize
    options= {'maxiter': 400}
    res = optimize.minimize(costFunction,
                            init_theta,
                            (X, y),
                            jac=True,
                            method='TNC',
                            options=options)
    cost = res.fun
    theta = res.x
    print(cost)
    print(theta)
    """


if __name__ == '__main__':
    plt.style.use('ggplot')
    main()