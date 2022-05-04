import pandas as pd
from scipy.io import arff
from sklearn.linear_model import LogisticRegression
import time
import GaussianNaiveBayes as gb
from sklearn.model_selection import train_test_split
import pandas as pd
import math
from math import sqrt, exp, pi, pow
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn import datasets
import pickle
from sklearn.decomposition import PCA

#Load the MNIST dataset
from sklearn.datasets import fetch_openml
#mnist = fetch_openml('mnist_784')

#mnist = arff.loadarff(open('mnist_784.arff', 'r'))
#print("File loaded")

df = pd.read_csv("mnist.csv")

X = df.drop(columns=['class'])
y = df['class']

print('data set loaded')

def load_iris():
    iris = datasets.load_iris()
    X = pd.DataFrame(iris['data'], columns=['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width'])
    y = pd.Series(iris['target'])
    return X, y

#X, y = load_iris()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

pca = PCA(n_components=180)
pca = pca.fit(X=X_train)
X_train_pca = pca.fit_transform(X_train)
X_train_pca = pd.DataFrame(X_train_pca)

X_test_pca = pca.transform(X_test)
X_test_pca = pd.DataFrame(X_test_pca)

print(pca.explained_variance_ratio_.sum())

enter_time = time.monotonic()
model = LogisticRegression(solver='lbfgs', max_iter=1000, random_state=42)
model.fit(X=X_train_pca, y=y_train)
exit_time =  time.monotonic()


y_pred = model.predict(X_test_pca)

print(confusion_matrix(y_test, y_pred))
print(precision_score(y_test, y_pred, average='micro'))
print("Sci Kit Lib: Number of mislabeled points out of a total %d points : %d" % (X_test.shape[0], (y_test != y_pred).sum()))

print("Stop watch = " + str(exit_time - enter_time))

enter_time = time.monotonic()
gnb_clf = gb.GaussianNaiveBayes(X_train_pca, y_train)

y_pred = gnb_clf.predict(X_test_pca)
exit_time =  time.monotonic()
print("Stop watch = " + str(exit_time - enter_time))

print(confusion_matrix(y_test, y_pred))
print(precision_score(y_test, y_pred, average='micro'))
print("my lib: Number of mislabeled points out of a total %d points : %d" % (X_test.shape[0], (y_test != y_pred).sum()))