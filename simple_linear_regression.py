import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# reading the txt file
df = pd.read_csv("data.txt", header=None, delimiter=",")


# sepreating features and labels
x = df[:][0]
y = df[:][1]


# function for gradient descent, taking features,labels, weights and learning rate as an input
def gradient_descent(x, y, w, learning):
    n = np.size(x)
    der_w0 = 0
    der_w1 = 0
    iterations = 7000

    # iterating till we get very small stepsize for weights which denotes we are close to minima of that function
    for i in range(iterations):

        # calculating the partial derivatives of each weights
        for i in range(n):
            der_w0 += -(2) * ((y[i] - (w[0] * x[i] + w[1])) * x[i])
            der_w1 += -(2) * (y[i] - (w[0] * x[i] + w[1]))

        # calculating the step size of each weights
        step_size_w0 = (der_w0 * learning) / n
        step_size_w1 = (der_w1 * learning) / n

        # calculating new weights
        new_w0 = w[0] - step_size_w0
        new_w1 = w[1] - step_size_w1
        w[0] = new_w0
        w[1] = new_w1

    return new_w0, new_w1


# calculating error function
# # def Error(x, y, y_new):
# #     error = []
# #     for i in range(x.shape[0]):
# #         e = (1 / x.shape[0]) * np.sum(np.square(y_new[i] - y[i]))
# #         error.append(e)
# #     return error

# initializing some random weights
w = [0, 1]

# calculating new weights by using gradient descent algorithm
weight = gradient_descent(x, y, w, learning=0.0000000096)

# initializing empty list to calculate predicted values of y with the help of new wieghts
y_new = []

# predicting new values of y using new weights
y_new = weight[1] * x + weight[0]

# ploting our regression line
plt.plot(x, y_new, color="red")

# ploting our initial datapoints
plt.scatter(x, y)
plt.show()
