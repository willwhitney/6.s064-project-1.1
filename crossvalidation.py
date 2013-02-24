import numpy as np
import perceptron
from project1 import *

def ptron(X, Y):
    return perceptron.perceptron(X, Y, 10)

def CV(alg, X, Y, k):
    

print CV(ptron, X1, Y1, 5)