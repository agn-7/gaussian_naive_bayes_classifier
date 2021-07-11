import numpy as np

from sklearn.naive_bayes import GaussianNB


X_train = np.array([
    [0, 18, 9.2, 8.1, 2, 1], [2, 17, 9.1, 9, 1.95, 1], [4, 16, 9, 10, 2.1, 1],
    [1, 20.1, 17, 15.5, 5, 0], [3, 23.5, 20, 20, 6.2, 0], [0, 21, 16.7, 16, 3.3, 0],
])
X_test = np.array([
    [1, 22, 15, 12, 4.5, 0]
])

y_train = np.array([0, 0, 0, 1, 1, 1])
gnb = GaussianNB()
y_pred = gnb.fit(X_train, y_train).predict(X_test)
print(y_pred)
