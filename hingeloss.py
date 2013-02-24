import numpy as np
from project1 import *

def nu(epoch):
    return 1 / (1.0 + epoch)

def epoch(updates, n):
    return int(updates / n)

def hlsgd(X, Y, l, nextIndex, numEpochs):
    n = Y.size
    theta = np.zeros(X[1].size)
    thetas = [theta]

    updates = 0
    while epoch(updates, n) < numEpochs:
        i = nextIndex(n)

        updateContribution = 0
        if Y[i] * np.dot(theta, X[i]) <= 1:
            updateContribution = Y[i] * X[i]

        thetaBar = np.concatenate(([0], theta[1:]))
        # print theta
        # print thetaBar

        update = l * thetaBar - updateContribution
        theta = theta - nu(epoch(updates, n)) * (update)

        # print theta
        thetas.append(theta)
        updates += 1

    options = []
    for theta in thetas:
        thetaBar = np.concatenate(([0], theta[1:]))
        regularization = 0.5 * l * (np.linalg.norm(thetaBar) ** 2)
        lossContribution = 0
        for i, row in enumerate(X):
            lossContribution += max(1 - Y[i] * np.dot(theta, X[i]), 0)
        
        loss = float(regularization + lossContribution / n)
        options.append((loss, theta))

    options.sort(key=lambda option: option[0])
    result = np.array(options[0][1])[np.newaxis]
    # print result.T
    return result.T


hlsgd(X1,Y1,0.01,CountingIndex().fn, 10)
