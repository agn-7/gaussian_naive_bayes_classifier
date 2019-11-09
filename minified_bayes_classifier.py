import numpy as np
from math import sqrt, pi


def gaussian(x, mu, sig):
    return 1./(sqrt(2.*pi)*sig)*np.exp(-np.power((x - mu)/sig, 2.)/2)


data_set = np.array([
    [1.14, 0], [1.22, 0], [1.18, 0], [1.34, 0], [1.24, 0],
    [2.12, 1], [2.36, 1], [1.86, 1], [1.64, 1], [2.92, 1]
])


y_probability_1 = len(data_set[:5, 1]) / (len(data_set[:5, 1]) + len(data_set[5:, 1]))
y_probability_2 = len(data_set[5:, 1]) / (len(data_set[:5, 1]) + len(data_set[5:, 1]))
likelihood = '''Suppose is a gaussian distribution.'''
x_probability = '''Marginal probability which is same in both classes.'''

mean1 = np.mean(data_set[:5, 0])
std1 = np.std(data_set[:5, 0])
variance1 = np.square(std1)  # pow(x, 2)

mean2 = np.mean(data_set[5:, 0])
std2 = np.std(data_set[5:, 0])
variance2 = np.square(std2)  # pow(std2, 2)


def test(i):
    if gaussian(x=i, mu=mean1, sig=std1) * y_probability_1 > \
            gaussian(x=i, mu=mean2, sig=std2) * y_probability_2:
        print("{} belongs to the 0 class".format(i))
    else:
        print("{} belongs to the 1 class".format(i))


def data_set_test():
    for i in data_set[:, 0]:
        test(i)


def live_test():
    while True:
        val = input("Enter a value to classify: ")
        test(float(val))


data_set_test()
live_test()
