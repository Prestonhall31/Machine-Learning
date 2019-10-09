# Udacity Data Analyst Nanodegree
## P8: Introduction to Machine Learning
---
October 2, 2019
Preston Hall


## Context

This repo hold my machine learning projects that I have completed or currently working on through Udacity's Data Analysis Nanodegree program. 

Machine learning algorithms 

### Naive Bayes
> A family of simple ["probabilistic classifiers"](https://en.wikipedia.org/wiki/Probabilistic_classification) based on applying Bayes theorem with strong independence assumptions between features. 

###### Importing Naive Bayes from Scikit
 ```python
from sklearn.naive_bayes import GaussianNB

clf = GaussianNB() 
clf.fit(features_train, labels_train)    
pred = clf.predict(features_test)
accuracy = clf.score(features_test, labels_test)

 ```

 ### Support Vector Machines (SVM)

 > Support-vector machines (SVMs, also support-vector networks) are [supervised learning](https://en.wikipedia.org/wiki/Supervised_learning) models with associated learning algorithms that analyze data used for classification and regression analysis. 

 ###### Importing SVC from Scikit
```python
from sklearn.svm import SVC

clf = SVC(kernel="rbf", C=10000, gamma='scale')
clf.fit(features_train, labels_train)    
pred = clf.predict(features_test)
accuracy = clf.score(features_test, labels_test)
```
