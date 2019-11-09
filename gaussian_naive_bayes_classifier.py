import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
from math import sqrt, pi


def gaussian(x, mu, sig):
    return 1./(sqrt(2.*pi)*sig)*np.exp(-np.power((x - mu)/sig, 2.)/2)


data_set = np.array([
    [1.14, 0], [1.22, 0], [1.18, 0], [1.34, 0], [1.24, 0],
    [2.12, 1], [2.36, 1], [1.86, 1], [1.64, 1], [2.92, 1]
])
categories = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
color1 = (0.69411766529083252, 0.3490196168422699, 0.15686275064945221, 1.0)
color2 = (0.65098041296005249, 0.80784314870834351, 0.89019608497619629, 1.0)
colormap = np.array([color1, color2])
plt.scatter(
    [data_set[:, 0]], np.zeros_like(np.arange(10)),
    c=colormap[categories],
    marker='o',
    alpha=0.9
)

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


min_, max_ = min(data_set[:5, 0]), max(data_set[:5, 0])
mu, sigma = mean1, std1
dist = stats.truncnorm((min_ - mu) / sigma, (max_ - mu) / sigma, loc=mu, scale=sigma)
values = dist.rvs(10000)
f = gaussian(values, mean1, sigma)
plt.plot(values, f)

min_, max_ = min(data_set[5:, 0]), max(data_set[5:, 0])
mu, sigma = mean2, std2
dist = stats.truncnorm((min_ - mu) / sigma, (max_ - mu) / sigma, loc=mu, scale=sigma)
values = dist.rvs(10000)
f = gaussian(values, mean2, sigma)
plt.plot(values, f, c='r')
plt.show()


plt.ylabel('gaussian distribution')
x = np.linspace(mean1 - 3*variance1, mean1 + 3*variance1, 100)
plt.plot(x, stats.norm.pdf(x, mean1, variance1))
plt.show()


x = np.linspace(mean2 - 3*variance2, mean2 + 3*variance2, 100)
plt.plot(x, stats.norm.pdf(x, mean2, variance2), c='r')
plt.show()


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
