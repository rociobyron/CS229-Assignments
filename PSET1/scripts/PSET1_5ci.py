"""
Smoothes provided spectra based on locally weighted linear regression.

@author Rocío Byron
created on 2017/10/19
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Import data in dataframe (indexes = wavelengths)
train = pd.read_csv("http://cs229.stanford.edu/ps/ps1/quasar_train.csv", )
test = pd.read_csv("http://cs229.stanford.edu/ps/ps1/quasar_test.csv")
tau = 5 # bandwidth parameter

def Weight(m, boundx, X, tau):
    if (m != len(X)):
        raise Exception('Dimensions not matching.')
    W = np.identity(m)

    for i in range(0, m):
        W[i, i] = 1 / 2 * np.exp(-np.power((boundx - X[i]) / tau, 2) / 2)

    return W

# DataFrame(numpy matrix)
Ftrain = pd.DataFrame(np.ones(train.shape), columns=train.columns)
Ftest = pd.DataFrame(np.ones(test.shape), columns=train.columns)

mtrain = Ftrain.shape[0]
mtest = Ftest.shape[0]
p = Ftrain.shape[1] #no wavelength evaluation points

# x values for linear regression fitting (both train and test)
X = pd.DataFrame({'x0': np.ones(p), 'x1': train.columns}, dtype='float64')
XT = np.transpose(X)

boundX = (X.x1).values
boundY = np.ones(boundX.shape)

for j in range(0, mtrain):
    print('Train sample {}/{}'.format(j, mtrain))

    Y = train.loc[j, :]

    for i in range(0, p):
        W = Weight(p, boundX[i], X.x1, tau)
        XTWX = np.dot(np.dot(XT, W), X)

        theta = np.dot(np.dot(np.dot(np.linalg.inv(XTWX), XT), W), Y.values)
        boundY[i] = theta[0] + theta[1]*boundX[i]

    if (j == mtrain - 1):
        plt.figure()

        plt.xlabel('Wavelength')
        plt.ylabel('Flux')
        plt.scatter(X.x1, Y.values, marker='o', color='r')
        plt.plot(boundX, boundY, color='b')
        plt.title(r'Weighted linear regression $\tau = {}$'.format(tau))
        plt.show()

    Ftrain.loc[j, :] = boundY

for j in range(0, mtest):
    print('Test sample {}/{}'.format(j, mtest))

    Y = test.loc[j, :]

    for i in range(0, p):
        W = Weight(p, boundX[i], X.x1, tau)
        XTWX = np.dot(np.dot(XT, W), X)

        theta = np.dot(np.dot(np.dot(np.linalg.inv(XTWX), XT), W), Y.values)
        boundY[i] = theta[0] + theta[1]*boundX[i]

    if (j == mtrain - 1):
        plt.figure()

        plt.xlabel('Wavelength')
        plt.ylabel('Flux')
        plt.scatter(X.x1, Y.values, marker='o', color='r')
        plt.plot(boundX, boundY, color='b')
        plt.title(r'Weighted linear regression $\tau = {}$'.format(tau))
        plt.show()

    Ftest.loc[j, :] = boundY

Ftrain.to_csv("./Ftrain.csv")
Ftest.to_csv("./Ftest.csv")
