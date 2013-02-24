import numpy as np
from project1 import *
def perceptron(X, Y, numEpochs):
    theta = np.zeros(X[1].size)
    theta0 = 0
    thetas = [(theta, theta0)]

    epochs = 0
    while epochs < numEpochs:
        missed = False
        for i, row in enumerate(X):
            comparator = np.dot(theta, row) + theta0
            if np.sign(comparator) != Y[i]:
                missed = True
                theta = theta + Y[i] * X[i]
                theta0 = theta0 + Y[i]
                thetas.append((theta, theta0))

            print thetas
        epochs += 1
        if missed == False:
            break



perceptron(X1, Y1, 2)