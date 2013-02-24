import numpy as np
import perceptron
from project1 import *

def CV(alg, X, Y, k):
    n = Y.size
    # print X
    # print Y
    # print "========================================"

    xFolds = np.array_split(X, k)
    yFolds = np.array_split(Y, k)
    print xFolds
    print yFolds
    
    misses = 0
    for i in xrange(k):
        xTraining = np.vstack(xFolds[:i] + xFolds[i + 1:])
        xTest = xFolds[i]

        yTraining = np.vstack(yFolds[:i] + yFolds[i + 1:])
        yTest = yFolds[i]

        print "=================== TRAINING ====================="
        print xTraining
        print yTraining

        print "=================== TEST ====================="
        print xTest
        print yTest
        theta = alg(xTraining, yTraining)
        for index, row in enumerate(xTest):
            comparator = np.dot(theta.T, row)
            if np.sign(comparator) != yTest[index]:
                misses += 1

    return misses / float(n)





def ptron(X, Y):
    return perceptron.perceptron(X, Y, 10)

print CV(ptron, X1, Y1, 2)