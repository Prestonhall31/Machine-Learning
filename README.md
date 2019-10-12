# Udacity Data Analyst Nanodegree
## P8: Introduction to Machine Learning
---
October 2, 2019
Preston Hall


## Context

This repo hold my machine learning projects that I have completed or currently working on through Udacity's Data Analysis Nanodegree program. 

## Machine learning algorithms 

### Naive Bayes
> A family of simple ["probabilistic classifiers"](https://en.wikipedia.org/wiki/Probabilistic_classification) based on applying Bayes theorem with strong independence assumptions between features. 

###### Importing Naive Bayes from Scikit

 ```python
from sklearn.naive_bayes import GaussianNB

clf = GaussianNB() 
clf.fit(X, Y)    
pred = clf.predict(x)
accuracy = clf.score(x, y)

 ```

 ### Support Vector Machines (SVM)

 > Support-vector machines (SVMs, also support-vector networks) are [supervised learning](https://en.wikipedia.org/wiki/Supervised_learning) models with associated learning algorithms that analyze data used for classification and regression analysis. 

![Support Vector Machine](images/support_vector_machine.png)

 ###### Importing SVC from Scikit

```python
from sklearn.svm import SVC

clf = SVC(kernel="rbf", C=10000, gamma='scale')
clf.fit(X, Y)    
pred = clf.predict(x)
accuracy = clf.score(x, y)
```

### Decision Tree Regression

> A decision tree is a decision support tool that uses a tree-like graph or model of decisions and their possible consequences, including chance event outcomes, resource costs, and utility. It is one way to display an algorithm that only contains conditional control statements.

![Decision Tree Graph](images/decision_tree.png)

**Entropy:** Controls how the DT decides where to split the data. measure of impuity in a bunch of examples. An Entropy of 1.0 is maximally impure state

**Biase-Variance dilemma:** The conflict in trying to simultaneously minimize these two sources of error that prevent supervised learning algorithms from generalizing beyond their training set: The bias error is an error from erroneous assumptions in the learning algorithm. 

#### Importing DecisionTree from Scikit

```python
from sklearn import tree

clf = tree.DecisionTreeClassifier(min_sample_split=2)
clf = clf.fit(X, Y)
pred = clf.predict(x)
accuracy = clf.score(x, y)
```

---

### Choose your own Algorithm

For this next mini-project, choose from the following algorithms to resaerch,  deploy, use it to make predictions, and evaluate results.

- **K Nearest Neighbor:** In pattern recognition, the k-nearest neighbors algorithm (k-NN) is a non-parametric method used for classification and regression. In both cases, the input consists of the k closest training examples in the feature space.
<br>

- **Adaboost:** Short for Adaptive Boosting, is a machine learning meta-algorithm formulated by Yoav Freund and Robert Schapire, who won the 2003 Gödel Prize for their work. It can be used in conjunction with many other types of learning algorithms to improve performance.
<br>

- **Random Forest:** Random forests or random decision forests are an ensemble learning method for classification, regression and other tasks that operates by constructing a multitude of decision trees at training time and outputting the class that is the mode of the classes or mean prediction of the individual trees.

I am going to use Random Forest from my Algorithm. 

 ###### Importing Random Forest from Scikit

```python
from sklearn.ensemble import RandomForestClassifier

clf = tree.RandomForestClassifier()
clf = clf.fit(X, Y)
pred = clf.predict(x)
accuracy = clf.score(x, y)
```
Optional parameters:
- n_estimators: number of trees in the forrest

The script ran very quick and smooth. Using the prettyPicture() output the following graph. 

![Random Forest](images/random_forest_results.png)

---

