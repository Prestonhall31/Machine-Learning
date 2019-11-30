#!/usr/bin/env python
# coding: utf-8

import sys
import pickle
import warnings
sys.path.append("../tools/")

# Local packages/files
from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

# Support packages
import numpy as np
import pandas as pd


### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "rb") as data_file:
    data_dict = pickle.load(data_file)


# import data_dict into a pandas dataframe for easier manipulation
df = pd.DataFrame.from_dict(data_dict, orient = 'index')


# Remove NaNs
df = df.replace('NaN', np.nan)

# create a list of the employees
names = pd.Series(list(data_dict.keys()))

# Rows TOTAL and THE TRAVEL AGENCY IN THE PARK are not people and so will be removed from the dataset. 
# The row 'LOCKHART EUGENE E' contains no information. 
df = df.drop(['TOTAL', 'THE TRAVEL AGENCY IN THE PARK', 'LOCKHART EUGENE E'])

df = df.drop(['long_term_incentive', 
              'deferred_income', 
              'deferral_payments', 
              'restricted_stock_deferred',
              'director_fees', 
              'loan_advances', 
              'email_address'], axis=1)


# ## Task 3: 
# Create new feature(s)

# Create and add features to the dataframe.
df['from_poi_with_shared_receipt_percentage'] = df.apply(lambda row: float(row.from_poi_to_this_person / 
                                                                           row.shared_receipt_with_poi), axis=1)

df['total_compensation'] = df.apply(lambda row: float(row.total_payments + 
                                                      row.total_stock_value), axis=1)



### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
# Setting emply features list for future use.
features_list = []

# Append features list
for col in df.columns:
    features_list.append(col)

# Move 'poi' to front of list
features_list.insert(0, features_list.pop(features_list.index('poi')))



### Store to my_dataset for easy export below.
my_dataset = df


# Replacing the np.nan values with the string 'NaN' for preprocessing.
my_dataset = my_dataset.replace(np.nan, 0)

# Converting data back to dict for sklearn manipulation.
my_dataset = my_dataset.to_dict('index')


### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)


# from sklearn.feature_selection import SelectKBest, f_classif
# # Create the object for SelectKBest and fit and transform the classification data
# # k is the number of features you want to select [here it's 2]

# X_clf_new=SelectKBest(score_func=f_classif,k=5).fit_transform(features,labels)


### Choose the 5 best features for testing
from sklearn.feature_selection import SelectKBest, f_classif

clf = SelectKBest(f_classif, k=5)
clf.fit_transform(features,labels)

# Print the resulting labels
k_best_lables = df.columns[clf.get_support(indices=True)]
k_best_lables_scores = clf.scores_[clf.get_support()]

labels_scores = list(zip(k_best_lables, k_best_lables_scores))
labels_scores_df = pd.DataFrame(data = labels_scores, columns=['Feat_names', 'F_Scores'])

#Sort the dataframe for better visualization
labels_scores_df_sorted = labels_scores_df.sort_values(['F_Scores', 'Feat_names'], ascending = [False, True])
# print(labels_scores_df_sorted)


### Use Sklearn's train_test_split model to separate data into testing and training data. 
from sklearn.model_selection import train_test_split
features_train, features_test, labels_train, labels_test =     train_test_split(features, labels, test_size=0.3, random_state=42)


## Naive Bayes
from sklearn.naive_bayes import GaussianNB
nb_clf = GaussianNB()

## Decision Tree
from sklearn import tree
dt_clf = tree.DecisionTreeClassifier(min_samples_split=30)

## Support Vector Machine Classifier
from sklearn.svm import SVC
svm_clf = SVC(gamma='scale', kernel='rbf', C=1000)

## Adaboost Classifier
from sklearn.ensemble import AdaBoostClassifier
ab_clf = AdaBoostClassifier(algorithm='SAMME.R')

## Random Forest
from sklearn.ensemble import RandomForestClassifier
rf_clf = RandomForestClassifier(n_estimators=10)


from sklearn import metrics
from sklearn.metrics import accuracy_score


clfs = [nb_clf, svm_clf, dt_clf, ab_clf, rf_clf]


def classifyer(clfs, f_train, f_test, l_train, l_test):

    classifier_names = ['Naive Bayes', 'Support Vector Machine', 'Decision Tree', 'Ada Boost', 'Random Forest']
    accuracy = []
    precision = []
    recall = []

    for classifier in [nb_clf, svm_clf, dt_clf, ab_clf, rf_clf]:

        clf = classifier
        clf.fit(f_train, l_train)
        predictions = clf.predict(f_test)

        accuracy.append("{0:.0%}".format(accuracy_score(l_test, predictions)))
        precision.append("{0:.0%}".format(metrics.precision_score(l_test, predictions)))
        recall.append("{0:.0%}".format(metrics.recall_score(l_test, predictions)))
        
    clf_df = pd.DataFrame([accuracy, precision, recall],
                          columns = classifier_names,
                          index=['Accuracy', 'Precision', 'Recall'])
    
    return clf_df



classifyer(clfs, features_train, features_test, labels_train, labels_test)


# ## Task 5: 
# Tune your classifier to achieve better than .3 precision and recall using our testing script. Check the tester.py script in the final project folder for details on the evaluation method, especially the test_classifier function. Because of the small size of the dataset, the script uses stratified shuffle split cross validation. For more info: http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html
# 


######################################################################################################
######################################################################################################



# Using K-Fold to validate 

from sklearn.model_selection import KFold
from sklearn import metrics

kf = KFold(n_splits = 4, shuffle = True)

for train_indices, test_indices in kf.split(labels):
    # make training and testing dataset
    kfeatures_train = [features[ii] for ii in train_indices]
    kfeatures_test = [features[ii] for ii in test_indices]
    klabels_train = [labels[ii] for ii in train_indices]
    klabels_test = [labels[ii] for ii in test_indices]


classifyer(clfs, kfeatures_train, kfeatures_test, klabels_train, klabels_test)


# ## Task 6: 
# Dump your classifier, dataset, and features_list so anyone can check your results. You do not need to change anything below, but make sure that the version of poi_id.py that you submit can be run on its own and generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)

