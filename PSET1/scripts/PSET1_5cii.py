import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def d(fA, FB):
    """
        Distance parameter
        Takes a [px1] ndarray and a [mxp] ndarray and returns the cumulative square distances
        between the first array and each of the rows of the matrix, [mx1]
    """

    p = np.shape(fA)[0]

    # convert FB to multidimensional if necessary
    if (len(np.shape(FB)) == 1):
        m = 1
        FB = FB.reshape(m, np.shape(FB)[0])
    else:
        m = np.shape(FB)[0]

    # input control
    if (p != np.shape(FB)[1] or len(np.shape(fA)) != 1):
        raise Exception("Mismatching sizes")

    d = np.empty(m)

    for i in range(0, m):
        delta = fA - FB[i, :]
        d[i] = np.dot(delta, delta)

    return d

def ker(t):
    return max(1-t, 0)


train = pd.read_csv("./Ftrain.csv", header=0, index_col=0)
mtrain = train.shape[0]

p = train.shape[1] # no wavelength evaluation points
x = train.columns.values.astype(float) # wavelengths, as a ND array of float

X = pd.DataFrame({'x0': np.ones(p), 'x1': x})

pright = np.argwhere(x >= 1300)[0][0] # first index >= 1300
pleft = np.argwhere(x >= 1200)[0][0] # first index >= 1200

# column partial indexes for left and right functions
col_right = train.columns[pright:p]
col_left = train.columns[0:pleft]

k = 3 # no. neighbours to be included in training and testing

# training of left function
fright_train = train.loc[:, col_right]
fleft_train = train.loc[:, col_left]
fleft_hat = np.empty(np.shape(fleft_train))
d_train = np.empty(np.shape(fleft_train.index))

for i in range(0, mtrain):
    d_right = d(fright_train.loc[i, :].values, fright_train.values) # finds distance between fright(i) and the rest
    d_sort = np.sort(d_right) # sorted distance array

    num = np.zeros(np.shape(fleft_train.loc[0, :])) # numerator of fleft_hat
    den = np.zeros(np.shape(fleft_train.loc[0, :])) # denominator of fleft_hat
    h = max(d_right) # maximum distance to function

    for j in range(1, k+1):
        # range excluding the first zero element (when d(fright(i), fright(i)) = 0)
        index = np.argwhere(d_right == d_sort[j])[0][0] # index of the jth closest neighbour
        num += ker(d_sort[j]/h) * fleft_train.loc[index, :]
        den += ker(d_sort[j]/h)

    fleft_hat[i, :] = np.divide(num, den)
    d_train[i] = d(fleft_hat[i, :], fleft_train.loc[i, :].values)

print(np.mean(d_train))

# test set
test = pd.read_csv("./Ftest.csv", header=0, index_col=0)
mtest = test.shape[0]

fright_test = test.loc[:, col_right]
fleft_test = test.loc[:, col_left]
d_test = np.empty(np.shape(fleft_test.index))

for i in range(0, mtest):
    d_right = d(fright_train.loc[i, :].values, fright_test.values) # finds distance between fright(i) and the test values
    d_sort = np.sort(d_right) # sorted distance array

    num = np.zeros(np.shape(fleft_test.loc[0, :])) # numerator of fleft_hat
    den = np.zeros(np.shape(fleft_test.loc[0, :])) # denominator of fleft_hat
    h = max(d_right) # maximum distance to function

    for j in range(1, k+1):
        # range excluding the first zero element (when d(fright(i), fright(i)) = 0)
        index = np.argwhere(d_right == d_sort[j])[0][0] # index of the jth closest neighbour
        num += ker(d_sort[j]/h) * fleft_train.loc[index, :]
        den += ker(d_sort[j]/h)

    fleft_hat[i, :] = np.divide(num, den)
    d_test[i] = d(fleft_hat[i, :], fleft_test.loc[i, :].values)


fig, ax = plt.subplots()
ax.plot(x[0:pleft], fleft_hat[0, :], 'r', label='Estimated fleft')
ax.plot(x, test.loc[0, :], 'b', label='Observed f')
ax.set_title('Example m = 1')
ax.set_xlim(1150, 1600)
ax.set_ylim(0.5, 1.5)

# Now add the legend with some customizations.
legend = ax.legend()
plt.show()

#
# fig, ax = plt.subplots()
# ax.plot(x[0:pleft], fleft_hat[0, :], 'r', label='Estimated fleft')
# ax.plot(x[0:pleft], fleft_test.loc[0, :], 'b', label='Observed fleft')
# ax.set_title('Example m = 1')
# ax.set_xlim(1150, 1200)
# ax.set_ylim(0, 2)
#
# # Now add the legend with some customizations.
# legend = ax.legend()
# plt.show()
#
# fig, ax = plt.subplots()
# ax.plot(x[0:pleft], fleft_hat[5, :], 'r', label='Estimated fleft')
# ax.plot(x[0:pleft], fleft_test.loc[5, :], 'b', label='Observed fleft')
# ax.set_title('Example m = 6')
# ax.set_xlim(1150, 1200)
# ax.set_ylim(0, 2)
#
# # Now add the legend with some customizations.
# legend = ax.legend()
# plt.show()

print(np.mean(d_test))
