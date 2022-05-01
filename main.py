import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
import pandas as pd
import math
import numpy as np
from scipy.stats import norm
from math import sqrt, exp, pi, pow
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
import seaborn as sb

# import some data to play with
def load_iris():
    iris = datasets.load_iris()
    X = pd.DataFrame(iris['data'], columns=['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width'])
    y = pd.DataFrame(iris['target'], columns=['type'])
    return X, y


def load_breast_cancer():
    bc = datasets.load_breast_cancer()
    X = pd.DataFrame(bc['data'], columns=bc['feature_names'])
    y = pd.DataFrame(bc['target'], columns=['malignant'])
    return X, y


class GaussianNaiveBayes:
    class ColumnGaussianAttributes:

        def __init__(self, col_name, mu, sigma, max_val, min_val):
            self.col_name = col_name
            self.mu = mu
            self.sigma = sigma
            self.max_val = max_val
            self.min_val = min_val

    def calc_maximum_likelihood(self, x, mu, std):
        # use the pdf to calculate a probability for each value
        # given the feature exhibits a Gaussian distribution
        # x = feature mean
        # mu = population mean
        # what is the probability that a feature = x, where does this lie in the dataset
        # distribution? Probabilities closer to the mean, will have more of an impact
        # we are using the priors to create a gaussian distribution in order
        # to calculate the present
        # This returns the likelihood

        return 1 / (std * sqrt(2 * pi)) * exp(-1 / 2 * pow(((x - mu) / std), 2))

    def __init__(self, train_X, train_y):

        # finds the gaussian distribution of each column

        # find the mean/std of each feature

        self.classifiers = {}
        self.prior_prob_list = {}

        # get the unique target values in y
        self.target_vals = pd.unique(y.iloc[0:, 0])
        total_train_size = len(train_X)

        # for each target value find the feature set std, mu and prior associated with that target value
        for val in self.target_vals:
            self.classifiers[val] = []

            # get all rows associated with the current target value
            clf_rows = train_X[train_y.iloc[0:, 0] == val]

            # store the prior for each target value
            self.prior_prob_list[val] = len(clf_rows) / total_train_size

            # get the std and mu of each column in the current list of rows
            for col in clf_rows:
                row_col = clf_rows[col]
                self.classifiers[val].append(self.ColumnGaussianAttributes(col,
                                                                           row_col.mean(),
                                                                           row_col.std(),
                                                                           row_col.max(),
                                                                           row_col.min()))

    def predict(self, X):

        prediction_list = []

        # Create a prediction for each row associated with each classification
        # the high association based on the PDF and Bayes' Theorem wins

        # Grab the current row to classify
        for index, row in X.iterrows():
            current_guess = None
            predicted_classifier = None

            # Get each classifier name from the classifier dict and calculate the probability of this row matching each
            # target classifier using the earlier sigma and mean calculations
            for clf_name in self.classifiers.keys():

                # Initial guess for classification of this row if it were to be this type of classification
                clf_probability = math.log(self.prior_prob_list[clf_name])

                # Grab column info list in the classifier dict and iterate through each column
                for gaussian_col_attr in self.classifiers[clf_name]:
                    # Get the column value from the current row and calculate the probability it belongs to
                    # this classifier based on the pdf and the prior
                    x = row[gaussian_col_attr.col_name]

                    # Calculate the rest of the probabilities based on X columns values
                    clf_probability += math.log(self.calc_maximum_likelihood(x,
                                                                             gaussian_col_attr.mu,
                                                                             gaussian_col_attr.sigma))

                # Check the current prediction and update it with the current calculations
                if current_guess is None or clf_probability > current_guess:
                    current_guess = clf_probability
                    predicted_classifier = clf_name

            prediction_list.append(predicted_classifier)

        return pd.Series(data=prediction_list)

X, y = load_breast_cancer()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

y_test = y_test.iloc[0:, 0].reset_index(drop=True)

mdl = GaussianNaiveBayes(X_train, y_train)

y_pred = mdl.predict(X_test).reset_index(drop=True)

print(confusion_matrix(y_test, y_pred))
print(precision_score(y_test, y_pred, average='micro'))
print("my lib: Number of mislabeled points out of a total %d points : %d" % (X_test.shape[0], (y_test != y_pred).sum()))

from sklearn.naive_bayes import GaussianNB

gnb = GaussianNB()
y_pred = gnb.fit(X_train, y_train.values.ravel()).predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(precision_score(y_test, y_pred, average='micro'))
print("scikit learn: Gaussian Number of mislabeled points out of a total %d points : %d" % (
    X_test.shape[0], (y_test != y_pred).sum()))
