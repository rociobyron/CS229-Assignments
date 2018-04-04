import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def dJxy(x, y):

    """ Returns the gradient vector of J(theta) given a set of (x, y) """
    # input control on x and y
    try:
        (m, nAmp) = x.shape # (samples, features) including row of "1"s
        my = y.shape[0]
    except:
        raise Exception("x and y must be numpy arrays")

    if (m != my):
        raise Exception("Dimensions of x and y not matching")

    def dJ(theta):

        # input control on theta
        try:
            nthAmp = len(theta)
        except:
            raise Exception("theta must be a numpy array")

        if (nAmp != nthAmp):
            raise Exception("Dimensions of x and theta not matching")

        # summation of samples
        summation = np.zeros(theta.shape)
        for i in range(0, m):
            xi = x[i, :]
            yi = y[i, :]
            zi = (yi * np.dot(theta, xi))

            summation = summation + 1 / (1 + np.exp(-zi)) * (-np.exp(-zi)) * (-yi * xi)

        return 1 / m * summation
    return dJ

def Hxy(x, y):

    """ Returns the Hessian matrix of J(theta) given a set of (x, y) """

    # input control on x and y
    try:
        (m, nAmp) = x.shape # (samples, features [including "1"s, hence "Amp"])
        my = y.shape[0]
    except:
        raise Exception("x and y must be a pandas object")

    if (m != my):
        raise Exception("Dimensions of x and y not matching")

    def H(theta):
        # input control on theta
        try:
            nthAmp = len(theta)
        except:
            raise Exception("theta must be a pandas object, list or array")

        if (nAmp != nthAmp):
            raise Exception("Dimensions of x and theta not matching")

        # summation of samples
        summation = np.zeros((nAmp, nAmp))
        for i in range(0, m):
            xi = x[i, :]
            yi = y[i, :]
            zi = (yi * np.dot(theta, xi))

            Ai = np.exp(-zi) / (1 + np.exp(-zi))
            Bi = np.power(yi, 2)
            Ci = (1 - Ai)

            summation = summation + Ai * Bi * Ci * np.dot(xi.reshape(-1, 1), xi.reshape(1, -1))

        return 1 / m * summation
    return H

def Newton(dJ, H, theta0, it=1e2):
    dJprev = dJ(theta0)
    Hprev = H(theta0)
    thprev = theta0

    count = 0 # while loop limited to 100 iterations by default
    error = np.linalg.norm(dJprev) # error as the 2-norm of the gradient

    while (error > 1e-3 and count < it):
        thnext = thprev + np.dot(np.linalg.inv(Hprev), dJprev)

        thprev = thnext
        dJprev = dJ(thprev)
        Hprev = H(thprev)
        error = np.linalg.norm(dJprev)
        count += 1

    return thprev

""" Application of functions """
x = pd.read_table("http://cs229.stanford.edu/ps/ps1/logistic_x.txt", delimiter="\s+",
                    header=None, names = ["x1", "x2"])
y = pd.read_table("http://cs229.stanford.edu/ps/ps1/logistic_y.txt", delimiter="\s+",
                    header=None, names=['y'])

# Plotting training set
fig, ax = plt.subplots()
plus_values = x[y['y'] > 0]
plus_values.plot.scatter(x='x1', y='x2', color='red', marker='+', label='y = 1',
                         ax=ax)

min_values = x[y['y'] < 0]
min_values.plot.scatter(x='x1', y='x2', color='black', marker='o', label='y = -1',
                        ax=ax)

# Newton's method
(m, n) = x.shape # (samples, true features)
x.insert(loc=0, column="x0", value=np.ones(m)) # additional feature x0 = 1
theta0 = np.zeros(n + 1)

dJ = dJxy(x.values, y.values)
H = Hxy(x.values, y.values)

theta = Newton(dJ, H, theta0)
print(theta)

# Plotting hypothesis
x1_sort = np.sort(x['x1'].values)
h = (-theta[0] - theta[1] * x1_sort) / theta[2]

plt.plot(x1_sort, h, label='h(x)', color='blue')
plt.show()
