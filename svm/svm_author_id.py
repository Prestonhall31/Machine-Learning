#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
import matplotlib as plt
<<<<<<< HEAD
sys.path.append("/Users/preston/Projects/DAND-MachineLearning/MachineLearningProjects/tools")
sys.path.append("/Users/preston/Projects/DAND-MachineLearning/MachineLearningProjects/choose_your_own")
=======
sys.path.append("/Users/preston/Projects/DAND-MachineLearning/ud120-projects/tools")
sys.path.append("/Users/preston/Projects/DAND-MachineLearning/ud120-projects/choose_your_own")
>>>>>>> 2a2c4ef95cd988759f41cd9f5eca5dc990424e84
from email_preprocess import preprocess
from class_vis import prettyPicture
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()




#########################################################
### your code goes here ###
# ftlen = int(len(features_train)/100)
# ltlen = int(len(labels_train)/100)

# features_train = features_train[:ftlen]
# labels_train = labels_train[:ltlen]

t0 = time() #Start time

clf = SVC(kernel="rbf", C=10000, gamma='scale') # classifier
clf.fit(features_train, labels_train)   # fit the training data  
pred = clf.predict(features_test) # predict using the features, not labels
print("training time:", round(time()-t0, 3), "s") # End time

# you can find a specific result using the following
print(pred[10])
print(pred[26])
print(pred[50])

# count the amount of occurances in prediction
chris = 0
sara = 0

for i in pred:
    if i == 1:
        chris += 1
    else:
        sara += 1

print("Chris: ", chris)
print("Sara: ", sara)

# Accuracy 
acc = accuracy_score(pred, labels_test)

print(acc)


#########################################################


