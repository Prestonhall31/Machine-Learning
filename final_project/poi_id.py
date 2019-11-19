#!/usr/bin/python

import sys
import pickle
from time import time
sys.path.append("../tools/")


from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data


### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi',
                 'salary', "to_messages",
                 "total_payments",
                 "exercised_stock_options",
                 "bonus",
                 "restricted_stock",
                 "shared_receipt_with_poi",
                 "restricted_stock_deferred",
                 "total_stock_value",
                 "expenses",
                 "loan_advances",
                 "from_messages",
                 "from_this_person_to_poi",
                 "director_fees",
                 "deferred_income",
                 "long_term_incentive",
                 "from_poi_to_this_person"]# You will need to use more features

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "rb") as data_file:
    data_dict = pickle.load(data_file)

### Task 2: Remove outliers

data_dict.pop('TOTAL', 0) # Contains column total data
data_dict.pop('THE TRAVEL AGENCY IN THE PARK', 0) # not an individual
data_dict.pop('LOCKHART EUGENE E', 0) # record contains no information

### Task 3: Create new feature(s)

### Store to my_dataset for easy export below.
my_dataset = data_dict

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

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# GaussianNB
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

clf = GaussianNB()
clf.fit(features, labels)    
pred = clf.predict(features)
acc = accuracy_score(pred, labels)

print("NB Accuracy: ", acc)


# Decision Tree
from sklearn import tree

clf = tree.DecisionTreeClassifier(min_samples_split=40)
clf = clf.fit(features, labels)
accuracy = clf.score(features, labels)

print("DT Accuracy: ", accuracy)


# Support Vector Machine Classifier
from sklearn.svm import SVC

clf = SVC(kernel="rbf", C=10000, gamma='scale') 
clf.fit(features, labels)   
accuracy = clf.score(features, labels)

print("SVC Accuracy: ", accuracy)


### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
from sklearn.model_selection import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)


from sklearn.model_selection import KFold
from sklearn import metrics

kf = KFold(n_splits=4, shuffle=True)

for train_index, test_index in kf.split(labels):
    kfeatures_train= [features[ii] for ii in train_index]
    kfeatures_test= [features[ii] for ii in test_index]
    klabels_train=[labels[ii] for ii in train_index]
    klabels_test=[labels[ii] for ii in test_index]

clf = clf.fit(kfeatures_train, klabels_train)
pred = clf.predict(kfeatures_test)

print("K-Fold Accuracy = ", accuracy_score(klabels_test, pred))
print('K-Fold Precision = ', metrics.precision_score(klabels_test, pred))
print('K-Fold Recall = ', metrics.recall_score(klabels_test, pred))


### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)