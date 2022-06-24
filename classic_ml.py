import numpy as np
import pandas as pd
import sklearn
import sklearn.svm
import sklearn.tree
import sklearn.ensemble
import sklearn.model_selection
import sklearn.metrics
import scipy
import scipy.stats


def svm_param_search(kernel, C, gamma, coef0, degree, n_iter, cv, verbose):
    '''
    Perform a randomized search on SVM hyperparameters using K-fold cross validation. The hyperparameter search
    is made for a specific SVM kernel defined in the input and it returns the best parameter configuration and
    accuracy for the corresponding kernel.
    '''

    svm = sklearn.svm.SVC(kernel=kernel)

    # define the parameter grid for each specific kernel
    if kernel == 'linear':
        param_distributions = {'C':C}
    elif kernel == 'poly':
        param_distributions = {'C':C, 'gamma':gamma, 'coef0':coef0, 'degree':degree}
    elif kernel == 'rbf':
        param_distributions = {'C':C, 'gamma':gamma}
    elif kernel == 'sigmoid':
        param_distributions = {'C':C, 'gamma':gamma, 'coef0':coef0}

    clf = sklearn.model_selection.RandomizedSearchCV(svm, param_distributions, n_iter=n_iter, cv=cv, verbose=verbose)

    clf.fit(X_train, y_train)

    return clf.best_estimator_, clf.best_params_, clf.best_score_


def svm_model_selection(kernels, C, gamma, coef0, degree, n_iter, cv, verbose):
    '''
    Select the best SVM model by performing a randomized search on hyperparameters using K-fold cross validation.
    The best SVM model is determined by the kernel and corresponding parameter configuration that obtained the highest
    value of accuracy on the hyperparameter search.
    '''

    print("\n")
    print("Selecting best SVM model")
    print("\n")

    accuracy = 0
    
    for i in kernels:
        print("SVM: (kernel=%s)" % i)
        best_estimator_, best_params_, best_score_ = svm_param_search(i, C, gamma, coef0, degree, n_iter, cv, verbose)
    
        if i == 'linear':
            print("Best Hyperparameters: (C=%.2f)" % best_params_['C'])
        elif i == 'poly':
            print("Best Hyperparameters: (C=%.2f, gamma=%.2f, coef0=%.2f, degree=%d)" % (best_params_['C'], best_params_['gamma'], best_params_['coef0'], best_params_['degree']))
        elif i == 'rbf':
            print("Best Hyperparameters: (C=%.2f, gamma=%.2f)" % (best_params_['C'], best_params_['gamma']))
        elif i == 'sigmoid':
            print("Best Hyperparameters: (C=%.2f, gamma=%.2f, coef0=%.2f)" % (best_params_['C'], best_params_['gamma'], best_params_['coef0']))
        
        print("Accuracy Score: %.1f%%" % (best_score_*100))
        print("\n")

        if best_score_ > accuracy:
            accuracy = best_score_
            best_estimator = best_estimator_
            best_kernel = i

    print("Best SVM Kernel: %s" % best_kernel)

    return best_estimator


def tree_model_selection(criterion, splitter, max_depth, n_iter, cv, verbose):
    '''
    Select the best Decision Tree model by performing a randomized search on hyperparameters using K-fold cross
    validation. The best Decision Tree model is determined by the parameter configuration that obtained the highest
    value of accuracy on the hyperparameter search.
    '''

    print("\n")
    print("Selecting best Decision Tree model")
    print("\n")

    param_distributions = {'max_depth':max_depth}

    accuracy = 0
    
    for i in criterion:
        for j in splitter:
            print("Decision Tree: (criterion=%s, splitter=%s)" % (i, j))

            tree = sklearn.tree.DecisionTreeClassifier(criterion=i, splitter=j)
            
            clf = sklearn.model_selection.RandomizedSearchCV(tree, param_distributions, n_iter=n_iter, cv=cv, verbose=verbose)
            
            clf.fit(X_train, y_train)

            print("Best Hyperparameters: (max_depth=%d)" % clf.best_params_['max_depth'])
            print("Accuracy Score: %.1f%%" % (clf.best_score_*100))
            print("\n")

            if clf.best_score_ > accuracy:
                accuracy = clf.best_score_
                best_estimator = clf.best_estimator_
                best_criterion = i
                best_splitter = j
            
    print("Best Decision Tree: (criterion=%s, splitter=%s)" % (best_criterion, best_splitter))

    return best_estimator


def forest_model_selection(criterion, n_estimators, max_depth, n_iter, cv, verbose):
    '''
    Select the best Random forest model by performing a randomized search on hyperparameters using K-fold cross
    validation. The best Random Forest model is determined by the parameter configuration that obtained the highest
    value of accuracy on the hyperparameter search.
    '''

    print("\n")
    print("Selecting best Random Forest model")
    print("\n")

    param_distributions = {'n_estimators':n_estimators, 'max_depth':max_depth}

    accuracy = 0
    
    for i in criterion:
        print("Random Forest: (criterion=%s)" % i)

        forest = sklearn.ensemble.RandomForestClassifier(criterion=i)
        
        clf = sklearn.model_selection.RandomizedSearchCV(forest, param_distributions, n_iter=n_iter, cv=cv, verbose=verbose)
        
        clf.fit(X_train, y_train)

        print("Best Hyperparameters: (n_estimators=%d, max_depth=%d)" % (clf.best_params_['n_estimators'], clf.best_params_['max_depth']))
        print("Accuracy Score: %.1f%%" % (clf.best_score_*100))
        print("\n")

        if clf.best_score_ > accuracy:
            accuracy = clf.best_score_
            best_estimator = clf.best_estimator_
            best_criterion = i
            
    print("Best Random Forest: %s" % best_criterion)

    return best_estimator


# prepare train data
ds_train = pd.read_csv('Datasets-Modified/dataset_train.csv', index_col=False)
ds_train.dropna(inplace=True) # why do we have NaN?
X_train = ds_train.drop(['label'], axis=1)
y_train = ds_train['label'].astype('int')

# prepare test data
ds_test = pd.read_csv('Datasets-Modified/dataset_test.csv', index_col=False)
ds_test.dropna(inplace=True) # why do we have NaN?
X_test = ds_test.drop(['label'], axis=1)
y_test = ds_test['label'].astype('int')

# global parameters
n_iter  = 20    # number of parameters sampled in randomized search
cv      = 3     # number of cross validation folds
verbose = 1

# svm specific parameters
kernels = ['linear', 'rbf', 'sigmoid'] # ['linear', 'poly', 'rbf', 'sigmoid']
c       = scipy.stats.reciprocal(1, 1000)
gamma   = scipy.stats.reciprocal(0.01, 10)
coef0   = scipy.stats.reciprocal(0.01, 10)
degree  = scipy.stats.randint(1, 10)

# decision tree specific parameters
criterion = ['gini', 'entropy']
splitter  = ['best', 'random']
max_depth = scipy.stats.randint(1, 100)

# random forest specific parameters
n_estimators = scipy.stats.randint(1, 100)

# get the best parameter configuration for each machine learning model
svm = svm_model_selection(kernels, c, gamma, coef0, degree, n_iter, cv, verbose)
tree = tree_model_selection(criterion, splitter, max_depth, n_iter, cv, verbose)
forest = forest_model_selection(criterion, n_estimators, max_depth, n_iter, cv, verbose)

# print performance metrics for each machine learning model
print("\n")
print("Accuracy for SVM model: %.1f%%" % (sklearn.metrics.accuracy_score(y_test, svm.predict(X_test))*100))
print("Precision for SVM model: %.1f%%" % (sklearn.metrics.precision_score(y_test, svm.predict(X_test))*100))
print("Recall for SVM model: %.1f%%" % (sklearn.metrics.recall_score(y_test, svm.predict(X_test))*100))
print("F1 Score for SVM model: %.1f%%" % (sklearn.metrics.f1_score(y_test, svm.predict(X_test))*100))
print("\n")
print("Accuracy for for Decision Tree model: %.1f%%" % (sklearn.metrics.accuracy_score(y_test, tree.predict(X_test))*100))
print("Precision for Decision Tree model: %.1f%%" % (sklearn.metrics.precision_score(y_test, tree.predict(X_test))*100))
print("Recall for Decision Tree model: %.1f%%" % (sklearn.metrics.recall_score(y_test, tree.predict(X_test))*100))
print("F1 Score for Decision Tree model: %.1f%%" % (sklearn.metrics.f1_score(y_test, tree.predict(X_test))*100))
print("\n")
print("Accuracy for for Random Forest model: %.1f%%" % (sklearn.metrics.accuracy_score(y_test, forest.predict(X_test))*100))
print("Precision for Random Forest model: %.1f%%" % (sklearn.metrics.precision_score(y_test, forest.predict(X_test))*100))
print("Recall for Random Forest model: %.1f%%" % (sklearn.metrics.recall_score(y_test, forest.predict(X_test))*100))
print("F1 Score for Random Forest model: %.1f%%" % (sklearn.metrics.f1_score(y_test, forest.predict(X_test))*100))
print("\n")

# print('\n')
# print(ds_train['label'].value_counts(dropna=False))
# print('\n')
# print(ds_test['label'].value_counts(dropna=False))
# print('\n')