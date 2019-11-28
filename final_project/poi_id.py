#!/usr/bin/python

import sys
import pickle
from time import time
sys.path.append("../tools/")

# Local packages/files
from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

# Support packages
import numpy as np
import pandas as pd

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "rb") as data_file:
    data_dict = pickle.load(data_file)

# import data_dict into a pandas dataframe for easier manipulation
df = pd.DataFrame.from_dict(data_dict, orient = 'index')

# import data_dict into a pandas dataframe for easier manipulation
df = pd.DataFrame.from_dict(data_dict, orient = 'index')

# Remove NaNs
df = df.replace('NaN', np.nan)

# create a list of the employees
names = pd.Series(list(data_dict.keys()))

### Task 2: Remove outliers

# These names are not people so will be removed
df = df.drop(['TOTAL', 'THE TRAVEL AGENCY IN THE PARK'])


# separate and count the NA values in poi
na_percentage = df.count() / len(df)
na_percentage.sort_values(ascending=False)

# separate and count the NA values in non_poi 
non_poi_df = df[df.poi == False]

# Removing these columns as there is more than 50% NaN's
df = df.drop(['long_term_incentive', 
              'deferred_income', 
              'deferral_payments', 
              'restricted_stock_deferred',
              'director_fees', 
              'loan_advances', 
              'email_address'], axis=1)


### Task 3: Create new feature(s)

# Create and add features to the dataframe.
df['from_poi_with_shared_receipt_percentage'] = df.apply(lambda row: float(row.from_poi_to_this_person / 
                                                                           row.shared_receipt_with_poi), axis=1)

df['total_compensation'] = df.apply(lambda row: float(row.total_payments + 
                                                      row.total_stock_value), axis=1)

# Append features list
features_list = []
for col in df.columns:
    features_list.append(col)

### Store to my_dataset for easy export below.
my_dataset = df

# Replacing the np.nan values with the string 'NaN' for preprocessing.
my_dataset = my_dataset.replace(np.nan, 'NaN')

# Converting data back to dict for sklearn manipulation.
my_dataset = my_dataset.to_dict('index')

### Extract features and labels from dataset for local testing

data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

### Use Sklearn's train_test_split model to separate data into testing and training data. 

from sklearn.model_selection import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

from sklearn import preprocessing
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif


min_max_scaler = preprocessing.MinMaxScaler()
fieatures_minmax = min_max_scaler.fit_transform(features)

selector = SelectKBest(score_func=f_classif, k=10)
features_transformed = selector.fit_transform(fieatures_minmax, labels)

### Choose the 5 best features for testing using SlectKBest
from sklearn.feature_selection import SelectKBest, f_classif

clf = SelectKBest(f_classif, k=5)
clf.fit_transform(features_train, labels_train)

# Print the resulting labels
k_best_lables = df.columns[clf.get_support(indices=True)]
k_best_lables_scores = clf.scores_[clf.get_support()]

labels_scores = list(zip(k_best_lables, k_best_lables_scores))
labels_scores_df = pd.DataFrame(data = labels_scores, columns=['Feat_names', 'F_Scores'])

#Sort the dataframe for better visualization
labels_scores_df_sorted = labels_scores_df.sort_values(['F_Scores', 'Feat_names'], ascending = [False, True])


### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# GaussianNB
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

clf = GaussianNB()
clf.fit(features_train, labels_train)    
pred = clf.predict(features_test)
acc = accuracy_score(pred, labels_test)


print("NB Accuracy: ", acc)


# Support Vector Machine Classifier
from sklearn.svm import SVC

clf = SVC(gamma='auto') # (kernel="rbf", C=10000, gamma='scale')
clf.fit(features_train, labels_train)  
pred = clf.predict(features_test) 
accuracy = accuracy_score(pred, labels_test)

print("SVC Accuracy: ", accuracy)


# Decision Tree
from sklearn import tree

clf = tree.DecisionTreeClassifier(min_samples_split=40)
clf.fit(features_train, labels_train)  
pred = clf.predict(features_test) 
accuracy = accuracy_score(pred, labels_test)

print("DT Accuracy: ", accuracy)


### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!





### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)