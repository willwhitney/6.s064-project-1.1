import numpy as np
from project1 import *

def perceptron(X, Y, numEpochs):
    theta = np.zeros(X[1].size)
    thetas = [theta]

    epochs = 0
    while epochs < numEpochs:
        missed = False
        for i, row in enumerate(X):
            comparator = np.dot(theta, row)
            if np.sign(comparator) != Y[i]:
                missed = True
                theta = theta + Y[i] * X[i]
                thetas.append(theta)

        epochs += 1
        if missed == False:
            result = np.array(theta)[np.newaxis]
            return result.T

    options = []
    for theta in thetas:
        misses = 0
        for i, row in enumerate(X):
            comparator = np.dot(theta, row)
            if np.sign(comparator) != Y[i]:
                misses +=1

        options.append((misses, theta))

    options.sort(key=lambda option: option[0])
    result = np.array(options[0][1])[np.newaxis]
    return result.T

perceptron(X1, Y1, 2)