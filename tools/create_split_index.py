import numpy as np

under = np.arange(1, 201)
upper = np.arange(201, 467)

np.random.shuffle(under)
np.random.shuffle(upper)

test_under = under[:50]
train_under = under[50:170]
val_under = under[170:]

test_upper = upper[:66]
train_upper = upper[66:226]
val_upper = upper[226:]

test = np.append(test_under, test_upper)
train = np.append(train_under, train_upper)
val = np.append(val_under, val_upper)

test.sort()
train.sort()
val.sort()

np.savetxt("test.csv", test, fmt='%d')
np.savetxt("train.csv", train, fmt='%d')
np.savetxt("val.csv", val, fmt='%d')