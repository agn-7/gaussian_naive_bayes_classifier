import numpy as np


def gaussian_uni(x, mu, sig):
    """
    Uni-variate normal distribution
    :param x: data (feature)
    :param mu: Mean
    :param sig: variance
    :return:
    """
    return 1./(np.sqrt(2.*np.pi)*sig)*np.exp(-np.power((x - mu)/sig, 2.)/2)


def gaussian_multi(x, d, mu, covariance):
    """
    pdf of the multivariate normal distribution.
    :param x: data (features)
    :param d: dimension
    :param mu: mean
    :param covariance: covariance
    :return:
    """
    try:
        x_m = x - mu
        return (1. / (np.sqrt((2 * np.pi)**d * np.linalg.det(covariance))) *
                np.exp(-(np.linalg.solve(covariance, x_m).T.dot(x_m)) / 2))
    except Exception as exc:
        print(exc)
        return 0


def multiply(*gaussians):
    """

    :param gaussians:
    :return:
    """
    result = 1
    for gau in gaussians:
        result *= gau

    return result


data_set = np.array([
    [0, 18, 9.2, 8.1, 2, 1], [2, 17, 9.1, 9, 1.95, 1], [4, 16, 9, 10, 2.1, 1],
    [1, 20.1, 17, 15.5, 5, 0], [3, 23.5, 20, 20, 6.2, 0], [0, 21, 16.7, 16, 3.3, 0],
])
categories = np.array([0, 0, 0, 1, 1, 1])

test_data = np.array([
    [1, 22, 15, 12, 4.5, 0]
])

y_probability_1 = len(data_set[:3]) / (len(data_set[:3]) + len(data_set[3:]))
y_probability_2 = len(data_set[3:]) / (len(data_set[:3]) + len(data_set[3:]))
likelihood = '''Suppose is a gaussian distribution.'''
x_probability = '''Marginal probability which is same in both classes.'''

mean1 = np.mean(data_set[:3], axis=0)
std1 = np.std(data_set[:3], axis=0)
variance1 = np.square(std1)  # pow(x, 2)
covariance1 = np.cov(np.stack(data_set[:3], axis=1))
# print(mean1)
# print(variance1)

mean2 = np.mean(data_set[3:], axis=0)
std2 = np.std(data_set[3:], axis=0)
variance2 = np.square(std2)  # pow(std2, 2)
covariance2 = np.cov(np.stack(data_set[3:], axis=1))

covariance_total = np.cov(np.stack(data_set, axis=0))
'''
Assuming that features are independently, we have to use variance not 
covariance and we can use the multiply of uni-variate gaussian distributions.
'''

# TODO
remove_zero = lambda i: 0.01 if i < 0.001 else i
'''plug a very small value instead of zero in order to avoid divided by zero'''
vfunc = np.vectorize(remove_zero)
variance1 = vfunc(variance1)
variance2 = vfunc(variance2)


def test(i):
    gaussians1 = gaussian_uni(x=i, mu=mean1, sig=variance1)
    gaussians2 = gaussian_uni(x=i, mu=mean2, sig=variance2)

    gaussians1 *= 10e5
    gaussians2 *= 10e5
    '''Normalization'''

    gaussians1 = vfunc(gaussians1)
    gaussians2 = vfunc(gaussians2)
    gaussian1 = np.prod(gaussians1)
    gaussian2 = np.prod(gaussians2)
    '''Multiplication gaussians together'''

    if gaussian1 * y_probability_1 > gaussian2 * y_probability_2:
        print("{} belongs to the class 0 which is Cat".format(i))
    else:
        print("{} belongs to the class 1 which is Dog".format(i))


def data_set_test():
    for i in data_set:
        test(i)


data_set_test()

print('Test Data:')
for i in test_data:
    test(i)
