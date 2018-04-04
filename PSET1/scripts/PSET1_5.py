import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Import data in dataframe (indexes = wavelengths)
train = pd.read_csv("http://cs229.stanford.edu/ps/ps1/quasar_train.csv", )
test = pd.read_csv("http://cs229.stanford.edu/ps/ps1/quasar_test.csv")

# # (b)i. Unweighted linear regression of the first example
# Y = train.loc[0, :]
# X = pd.DataFrame({'x0': np.ones(len(Y)), 'x1': Y.index}, dtype='float64')
#
# XT = np.transpose(X)
# XTX = np.dot(XT, X)
#
# theta = np.dot(np.dot(np.linalg.inv(XTX), XT), Y.values)
# print(theta)
#
# Xmax = X.max()[1]
# Xmin = X.min()[1]
# boundX = np.arange(Xmin, Xmax, (Xmax - Xmin)/100)
# boundY = theta[0] + theta[1]*boundX
# plt.xlabel('Wavelength')
# plt.ylabel('Flux')
# plt.scatter(X.x1, Y.values, marker='o', color='r')
# plt.plot(boundX, boundY, color='b')
# plt.title(r"[$\theta_0,\theta_1$] = {}".format(theta))
# plt.savefig('PSET1_5bi.png', bbox_inches='tight')

# # (b)ii. Weighted linear regression of the first example
# tau = 5

def Weight(m, boundx, X, tau):
    if (m != len(X)):
        raise Exception('Dimensions not matching.')
    W = np.identity(m)

    for i in range(0, m):
        W[i, i] = 1 / 2 * np.exp(-np.power((boundx - X[i]) / tau, 2) / 2)

    return W


Y = train.loc[0, :]
X = pd.DataFrame({'x0': np.ones(len(Y)), 'x1': Y.index}, dtype='float64')

XT = np.transpose(X)

Xmax = X.max()[1]
Xmin = X.min()[1]
boundX = np.arange(Xmin, Xmax, (Xmax - Xmin)/len(Y.values))
boundY = np.ones(boundX.shape)

for tau in [1, 10, 100, 1000]:
    for i in range(0, len(boundX)):
        W = Weight(len(Y.values), boundX[i], X.x1, tau)
        XTWX = np.dot(np.dot(XT, W), X)

        theta = np.dot(np.dot(np.dot(np.linalg.inv(XTWX), XT), W), Y.values)

        boundY[i] = theta[0] + theta[1]*boundX[i]

    plt.figure()

    plt.xlabel('Wavelength')
    plt.ylabel('Flux')
    plt.scatter(X.x1, Y.values, marker='o', color='r')
    plt.plot(boundX, boundY, color='b')
    plt.title(r'Weighted linear regression $\tau = {}$'.format(tau))
    plt.savefig('PSET1_5biii{}.png'.format(tau), bbox_inches='tight')
