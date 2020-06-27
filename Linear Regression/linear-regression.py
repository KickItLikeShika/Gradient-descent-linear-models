import numpy as np
import matplotlib.pyplot as plt


def final_plot(th, X, y):
    """Plotting the data and the best fit line after the optimization and check
    the results."""
    # Draw the graph to show the results
    plt.scatter(X[:, 1], y)
    plt.plot(X[:, 1], np.dot(X, th), 'blue')
    plt.show()



def computeHypothesis(X, theta):
    """ Compute H_theta(X) Theat(transopse) * X."""
    thetaT = theta.transpose()
    return X @ thetaT


def lossFunction(X, y, theta):
    """Computing the cost and how good our model was."""
    summation = 0
    m = y.size
    # Compute our predicted value
    hypothesis = computeHypothesis(X, theta)
    for i in range(m):
        squared_error = (hypothesis[i] - y[i]) ** 2
        summation += squared_error

    return summation / (2*m)



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
            hypothesis = computeHypothesis(X, theta)
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
        j = lossFunction(X, y, theta)
        # Print our loss to see our progress
        print("Cost Funciton: ", j)
        # Savin the loss function values in our list
        jh.append(j)

    return theta, jh


def first_plot(X, y):
    # Plot data
    plt.scatter(X, y)
    plt.show()


def main():
    # Load the data
    data = np.loadtxt('Data/ex1data1.txt', delimiter=',')
    # Split the data and get our features and our target
    X, y = data[:, 0], data[:, 1]
    first_plot(X, y)
    # The number of rows in the data set
    m = y.size
    # Insert the X0 columns
    X = np.stack([np.ones(m), X], axis=1)    
    # Initialize the parameters with zeros
    theta = np.zeros(2)
    # Check the loss function before any optimizataion
    ls = lossFunction(X, y, theta)
    print("The cost before the optimization: ", ls)
    # The learning rate
    alpha = 0.01
    iterations = 1500
    th, jh = gradientDescent(X, y, theta, alpha, iterations)
    # Display the results
    print("Gradient Descent: ", th)
    final_plot(th, X, y)
    fls = lossFunction(X, y, th)
    print("The cost after the optimization: ", fls)


if __name__ == '__main__':
    plt.style.use('ggplot')
    main()