import numpy as np
import pandas as pd
import sklearn
import sklearn.svm
import sklearn.tree
import sklearn.ensemble
import sklearn.metrics
import sklearn.model_selection
import scipy
import scipy.stats


ds_train = pd.read_csv('Datasets/dataset_train.csv', index_col=0)
ds_test = pd.read_csv('Datasets/dataset_test.csv', index_col=0)

X_train = ds_train.drop(['First_DC', 'Second_DC', 'Annotation'], axis=1)
y_train = ds_train['Annotation']

X_test = ds_test.drop(['First_DC', 'Second_DC', 'Annotation'], axis=1)
y_test = ds_test['Annotation']



# tree = sklearn.tree.DecisionTreeClassifier(random_state=0, max_depth=10)
# tree.fit(X_train,y_train)

# accuracy_train = sklearn.metrics.accuracy_score(y_train, tree.predict(X_train))
# accuracy_test = sklearn.metrics.accuracy_score(y_test, tree.predict(X_test))

# print("training accuracy: %.1f%%" % (accuracy_train*100))
# print("held-out accuracy (testing): %.1f%%" % (accuracy_test*100))



svc = sklearn.svm.SVC(kernel='rbf')
param_distributions = {'C':scipy.stats.reciprocal(1, 1000), 'gamma':scipy.stats.reciprocal(0.01, 10)}

clf = sklearn.model_selection.RandomizedSearchCV(svc,param_distributions, n_iter=100, cv=3, random_state=0, verbose=1)
clf.fit(X_train, y_train)

print("Best Hyperparameters: (C=%.2f, gamma=%.2f)" % (clf.best_params_['C'], clf.best_params_['gamma']))
print("Accuracy Score: %.1f%%" % (clf.best_score_*100))