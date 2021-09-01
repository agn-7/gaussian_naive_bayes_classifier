import numpy as np
import pandas as pd


data = [['black', 18, 9.2, 8.1, 2, 'True', 'cat'],
        ['orange', 17, 9.1, 9, 1.95, 'True', 'cat'],
        ['white', 16, 9, 10, 2.1, 'True', 'cat'],
        ['gray', 20.1, 17, 15.5, 5, 'False', 'dog'],
        ['brown', 23.5, 20, 20, 6.2, 'False', 'dog'],
        ['black', 21, 16.7, 16, 3.3, 'False', 'dog'],
        ]
df = pd.DataFrame(data, columns=['color', 'body length', 'height', 'weight',
                                 'ear length', 'claws', 'class'])

test_data = np.array([
    ['gray', 22, 15, 12, 4.5, 'False']
])
test = pd.DataFrame(test_data)


def categorical_posterior(x, y):
    """
    calculate posterior probability for each record
    :param x: feature
    :param y: class/label
    :return: probability
    """
    pass


def gaussian_uni(x, mu, sig):
    """
    Uni-variate normal distribution
    :param x: data (feature)
    :param mu: Mean
    :param sig: variance
    :return:
    """
    return 1./(np.sqrt(2.*np.pi)*sig)*np.exp(-np.power((x - mu)/sig, 2.)/2)


prior_probabilities = df['class'].value_counts()
y_probability1 = prior_probabilities['cat'] / prior_probabilities.sum()
y_probability2 = prior_probabilities['dog'] / prior_probabilities.sum()
'''Prior probabilities'''

likelihood = '''Supposed Gaussian (normal distribution) for numerical data.'''
x_probability = '''Marginal probability'''

mean1 = np.mean(df.loc[df['class'] == 'cat'], axis=0)
std1 = np.std(df.loc[df['class'] == 'cat'], axis=0)
variance1 = np.square(std1)  # pow(x, 2)

mean2 = np.mean(df.loc[df['class'] == 'dog'], axis=0)
std2 = np.std(df.loc[df['class'] == 'dog'], axis=0)
variance2 = np.square(std2)  # pow(std2, 2)
'''
Assuming that features are independently, we have to use variance not 
covariance and we can use the multiply of uni-variate gaussian distributions.
'''

numerical_df = df.drop(['color', 'claws'], axis=1)
categorical_df = df[['color', 'claws', 'class']]
'''Separation data by their data type.'''


def numerical_posterior(df):
    """

    :param df:
    :return:
    """
    posteriors1 = []
    posteriors2 = []
    for _, i in df.iterrows():
        posteriors1.append(gaussian_uni(x=i, mu=mean1, sig=variance1))
        posteriors2.append(gaussian_uni(x=i, mu=mean2, sig=variance2))

    return pd.DataFrame(posteriors1), pd.DataFrame(posteriors2)


def categorical_posterior(df):
    """

    :param df: Categorical data frame with no class.
    :return:
    """
    p1 = []
    p2 = []

    def x_probability(x):  # TODO
        return 1

    def likelihood(column, class_='class'):
        print(column, class_)
        df.groupby([column, class_]).count() / prior_probabilities[class_]
        print(222222222222)

    for column in df:
        for _, features in categorical_df.loc[
            categorical_df['class'] == 'cat'
        ].iterrows():
            p1.append((likelihood(column, 'cat') * y_probability1)
                      / x_probability(features[column]))

    for column in df:
        for _, features in categorical_df.loc[
            categorical_df['class'] == 'dog'
        ].iterrows():
            p2.append((likelihood(column, 'cat') * y_probability2)
                      / x_probability(features[column]))

    return pd.DataFrame(p1), pd.DataFrame(p2)

# def test(i):
#     gaussians1 = gaussian_uni(x=df.iloc[0], mu=mean1, sig=variance1)
#     gaussians2 = gaussian_uni(x=i, mu=mean2, sig=variance2)
#
#     gaussians1 *= 10e5
#     gaussians2 *= 10e5
#     '''Normalization'''
#
#     gaussian1 = np.prod(gaussians1)
#     gaussian2 = np.prod(gaussians2)
#     '''Multiplication gaussians together'''
#
#     if gaussian1 * y_probability_1 > gaussian2 * y_probability_2:
#         print("{} belongs to the class 0 which is Cat".format(i))
#     else:
#         print("{} belongs to the class 1 which is Dog".format(i))

X_num = numerical_df.drop(['class'], axis=1)
X_cat = categorical_df.drop(['class'], axis=1)
num_post1, num_post2 = numerical_posterior(X_num)
cat_post1, cat_post2 = categorical_posterior(categorical_df)


print('Test Data:')
for i in test:
    test(i)
