#!/usr/bin/python

import sys
import pickle
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
# data_dict.pop('LOCKHART EUGENE E', 0) # record contains no information
# data_dict.pop('HUMPHREY GENE E', 0) # 'to_poi_rate' outlier
# data_dict.pop('LAVORATO JOHN J', 0) # 'from_poi_to_this_person' / 'total_payments' outlier
# data_dict.pop('FREVERT MARK A', 0) # 'total_payments' outlier

### Task 3: Create new feature(s)
my_dataset = {}
for key in data_dict:
    my_dataset[key] = data_dict[key]
    try:
        from_poi_rate = 1. * data_dict[key]['from_poi_to_this_person'] / \
        data_dict[key]['to_messages']
    except:
        from_poi_rate = "NaN"
    try:
        to_poi_rate = 1. * data_dict[key]['from_this_person_to_poi'] / \
        data_dict[key]['from_messages']
    except:
        to_poi_rate = "NaN"
    my_dataset[key]['from_poi_rate'] = from_poi_rate
    my_dataset[key]['to_poi_rate'] = to_poi_rate

### Store to my_dataset for easy export below.
my_dataset = data_dict

### Extract features and labels from dataset for local testing
#data = featureFormat(my_dataset, features_list, sort_keys = True)

data = featureFormat(my_dataset, features_list, remove_NaN=True, remove_all_zeroes=True, 
                     remove_any_zeroes=True, sort_keys=True)
labels, features = targetFeatureSplit(data)


### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()

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

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)