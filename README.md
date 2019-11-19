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


dot_data = tree.export_graphviz(clf, out_file=None, 
                     feature_names=iris.feature_names,  
                     class_names=iris.target_names,  
                     filled=True, rounded=True,  
                     special_characters=True)  
graph = graphviz.Source(dot_data)  
graph 

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

## Exploring Enron Corpus Dataset

Types of data you can encounter when using ML
- Numerical: Numerical values (numbers)
- Categorical: Limited number of discrete values (category)
- Time Series: Temporal Value (Date, Timestamp)
- Text: Words (can be converted to numbers)

---

### LinearRegression

In statistics, linear regression is a linear approach to modeling the relationship between a scalar response (or dependent variable) and one or more explanatory variables (or independent variables). The case of one explanatory variable is called simple linear regression. For more than one explanatory variable, the process is called multiple linear regression.


```python
from sklearn.linear_model import LinearRegression
### reg = Regression
reg = LinearRegression()
reg.fit(X, y) 
reg.score(X, y) # returns the R-squared score. Compare R-squared test to R-squared training data.
reg.predict([]) # Takes a list of at least one item and predicts the outcome. 
reg.intercept_ # returns the y-intercept
reg.coef_ # returns the slope


```

R-squared: A statistical measure that represents the proportion of the variance for a dependent variable that's explained by an independent variable or variables in a regression model. 

##### Linear Regression Errors
The actual value vs the predicted value. 

- ###### Sum of Squares Error
Minimizing the sum of squares error (SSE)
Ordinary Least Squares (OLS) - What SKlearn uses
Gradient Descent - (not used in this class)

* There can be mulitple lines that minimize $\sum|error|$, but only one line will minimize $\sum|error^2|$. This is why we use the squared sum. 
Using SSE makes the implementation much easier as well. 

Problems with SSE: Typically Large SSE equal a worse fit. But great data points, which is usually better, will increase the SSE. Makes it hard to compare two different sets of data. 

- ###### $ r^2 $ ("r squared") of a regression
Does not have the same shortcoming as SSE. 

Returns a value  $ 0.0 < r^2 < 1.0 $ where 0.0 is not really capturing the data well.

Independant from the number of training points. 

> Note that R2 is only bounded from below by 0 when evaluating a linear regression on its training set. If evaluated on a significantly different set, where the predictions of a regressor are worse than simply guessing the mean value for the whole set, the calculation of R2 can be negative.

View the graph with Matplotlib
```python

plt.scatter(x_value, y_value)
plt.plot(x_value, reg.predict(x_value), color='blue', linewidth=3)
plt.xlabel("Title")
plt.ylabel("Other Title")
plt.show()
```

##### Classification vs Regression

| Property | Supervised Classification | Regression|
|---|---|---|
| Output Type | Discreet (class labels) | continuous (number) | 
| What are you trying to find? | decision (boundary) |  "best fit" line |
| Evaluation | accuracy | SSE or $ r^2 $ (r squared) |

##### Multivariate Regressions

A method used to measure the degree at which more than one independent variable (predictors) and more than one dependent variable (responses), are linearly related.

### Outliers

There is data that is going to pull the regression and lower the score. 
Remove all NaN's and outliers

# Unsupervisoed Machine Learning


k-means: k-means clustering is a method of vector quantization, originally from signal processing, that is popular for cluster analysis in data mining. Works in 2 stpes
1. Assign
2. Optimize
3. Repeat

[Visual Aid](https://www.naftaliharris.com/blog/visualizing-k-means-clustering/)

[Sklearn KMeans Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html#sklearn.cluster.KMeans)


### Feature Scaling

Feature scaling is a method used to normalize the range of independent variables or features of data. In data processing, it is also known as data normalization and is generally performed during the data preprocessing step.


$ x^1 = \frac{x - x_{min}}{x_{max} - x_{min}} $

Scaling features to a range using minmaxScaling with sklearn


from sklearn.preprocessing import MinMaxScaler
import numpy as np
X_train = np.array([[ 1., -1.,  2.],
                   [ 2.,  0.,  0.],
                    [ 0.,  1., -1.]])

scaler = MinMaxScaler()
rescaled_X_train = scaler.fit_transform(X_train)
X_scaled                                          


### Dimensions when Learning From Text


Bag of words: The bag-of-words model is a simplifying representation used in natural language processing and information retrieval. In this model, a text is represented as the bag of its words, disregarding grammar and even word order but keeping multiplicity. The bag-of-words model has also been used for computer vision.

Low-information words: words that appear in almost every set of text and so doesn't mean anything. Stop words are words that need to be removed completely as it appears too frequently, like 'the', 'in', 'a' and 'and'.

NLTK - National Language Tool Kit. 

To use stopwords for this first time, download the corpus. 

```python
import nltk

nltk.download()
```

```python
from nltk.corpus import stopwords

sw = stopwords.words("english")
```

There is a way to bundle similar words to count as one word. Use a stemmer to do this. 
e.g. [responsive, respond, responsive, repond] -> respon

#### Stemming with NLTK

```python
from nltk.stem.snowball import SnowballStemmer
# There are several types of stemmers

stemmer = SnowballStemmer("english")

#examples
stemmer.stem("Responsiveness")
>>> u'respon'
stemmer.stem('responsitivity')
>>>u'respon'
stemmer.stem('unresponsive')
>>>u'unrespon'
# there are certain limitations to this type of stemmer. This might be good or bad depending on what you're looking for. 
```

##### TfIdf Frequency

Tf = Term Frequency (like bag-of-words)
Idf = Inverse Document Frequency (the word gets a weighting to words that have a rare occurance)




## Feature Selection

There are several go-to methods of automatically selecting your features in sklearn. Many of them fall under the umbrella of univariate feature selection, which treats each feature independently and asks how much power it gives you in classifying or regressing.

There are two big univariate feature selection tools in sklearn: SelectPercentile and SelectKBest. The difference is pretty apparent by the names: SelectPercentile selects the X% of features that are most powerful (where X is a parameter) and SelectKBest selects the K features that are most powerful (where K is a parameter).

A clear candidate for feature reduction is text learning, since the data has such high dimension. We actually did feature selection in the Sara/Chris email classification problem during the first few mini-projects; you can see it in the code in tools/email_preprocess.py .


Bias-Variance Dilemma and No. of Features

High Bias - pays little attention to data oversimplication
     - High error on training set
High variance - pays too much attention to data (does not generalize well) overfits 
     - Much higher error on test set than on training set

How many features so that it falls in the middle of these two. 

THis is called regularization

Using Lasso for regression in sklearn

```python 
import sklearn.linear_model.Lasso

features, lables = GetMyData()
regression = Lasso()
regression.fit(features, labels)
regression.predict([2,4])
print(regression.coeff_)  # any 0.0 values means it will not use that feature
```

#### PCA: Principle Component Analysis

 

[PCA](https://en.wikipedia.org/wiki/Principal_component_analysis) is a statistical procedure that uses an orthogonal transformation to convert a set of observations of possibly correlated variables (entities each of which takes on various numerical values) into a set of values of linearly uncorrelated variables called principal components. This transformation is defined in such a way that the first principal component has the largest possible variance (that is, accounts for as much of the variability in the data as possible), and each succeeding component in turn has the highest variance possible under the constraint that it is orthogonal to the preceding components. The resulting vectors (each being a linear combination of the variables and containing n observations) are an uncorrelated orthogonal basis set. PCA is sensitive to the relative scaling of the original variables.

 

![PCA](https://en.wikipedia.org/wiki/Principal_component_analysis#/media/File:GaussianScatterPCA.svg)


When implementing PCA, it seems like it creates a regression line but it is actually turing 2D data graph into 1 dimensional. 

variance - technical term in statistics, roughly the spread of data distribution(similar to the standard deviation)

Using PCA to determine the maximum variance

```python
def doPCA:
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    pca.fit(data)
    return pca

pca = doPCA()
print(pca.explained_variance_ratio_)
first_pc = pca.components_[0]
second_pc = pca.components_[1]

transformed_data = pca.transform(data)
for ii, jj in zip(transformed_data, data):
    plt.scatter( first_pc[0]*ii[0], first_pc[1]*ii[0],color='r' )
    plt.scatter( second_pc[0]*ii[1], first_pc[1]*ii[1],color='c' )
    plt.scatter( jj[0], jj[1], color='b' )

```

When to use PCA
- latent features driving the patterns in the data
- dimensional reduction 
    - visual high-dimensional data
    - reduce noise
    - make other algorithms (regression, classification)
        work better b/c fewer inputs



### Splitting your data between training and testing data


```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42)
```
This code will split the data into 4 sections, training and testing data for features(X), and training and testing data for labels(Y)

Where to use training vs. testing data 

train/test split  ->  PCA  ->  SVM

pca.fit(training_features)
pca.transform(training_features)
svc.train(training_features)
_______________________________
pca.transform(test_features)
svc.predict(test_features)


K-fold Cross validation
Cross-validation is a resampling procedure used to evaluate machine learning models on a limited data sample. The procedure has a single parameter called k that refers to the number of groups that a given data sample is to be split into.

```python
from sklearn.model_selection import KFold

kf = KFold(n_splits=2)

for train_index, test_index in kf.split(X):
   print("TRAIN:", train_index, "TEST:", test_index)
   X_train, X_test = X[train_index], X[test_index]
   y_train, y_test = y[train_index], y[test_index]

```


### Grid Search

GridSearchCV is a way of systematically working through multiple combinations of parameter tunes, cross-validating as it goes to determine which tune gives the best performance. The beauty is that it can work through many combinations in only a couple extra lines of code.

Here's an example from the sklearn documentation:

```parameters = {'kernel':('linear', 'rbf'), 'C':[1, 10]}```
```svr = svm.SVC()```
```clf = grid_search.GridSearchCV(svr, parameters)```
```clf.fit(iris.data, iris.target)```

Let's break this down line by line.

```parameters = {'kernel':('linear', 'rbf'), 'C':[1, 10]}```
A dictionary of the parameters, and the possible values they may take. In this case, they're playing around with the kernel (possible choices are 'linear' and 'rbf'), and C (possible choices are 1 and 10).

Then a 'grid' of all the following combinations of values for (kernel, C) are automatically generated:

```('rbf', 1)	('rbf', 10)```
```('linear', 1)	('linear', 10)```

Each is used to train an SVM, and the performance is then assessed using cross-validation.

```svr = svm.SVC()```

This looks kind of like creating a classifier, just like we've been doing since the first lesson. But note that the "clf" isn't made until the next line--this is just saying what kind of algorithm to use. Another way to think about this is that the "classifier" isn't just the algorithm in this case, it's algorithm plus parameter values. Note that there's no monkeying around with the kernel or C; all that is handled in the next line.

```clf = grid_search.GridSearchCV(svr, parameters)```
This is where the first bit of magic happens; the classifier is being created. We pass the algorithm (svr) and the dictionary of parameters to try (parameters) and it generates a grid of parameter combinations to try.

```clf.fit(iris.data, iris.target)```
And the second bit of magic. The fit function now tries all the parameter combinations, and returns a fitted classifier that's automatically tuned to the optimal parameter combination. You can now access the parameter values via clf.best_params_.


### Confusion matrix

A confusion matrix is a table that is often used to describe the performance of a classification model (or “classifier”) on a set of test data for which the true values are known. It allows the visualization of the performance of an algorithm.


![Confusion Matrix](images/confusion_matrix.png)


Recall: True Positive / (True Positive + False Negative). Out of all the items that are truly positive, how many were correctly classified as positive. Or simply, how many positive items were 'recalled' from the dataset.

Precision: True Positive / (True Positive + False Positive). Out of all the items labeled as positive, how many truly belong to the positive class.


### Maching learning steps

#### Dataset/Questions
1. Do I have enough data?
2. Can I define a question?
3. enough/right features to answer questions?

#### Features
1. Exploration
a. Inspect for corrections
b. outlier removal 
c. imputation 
d. cleaning
2. Creation
a. Think about it like a human
3. Representation
a. Text vectorization
b. discretization
4. Scaling
a. mean subtraction
b. minmax scaler
c. standard scaler
5. Selection
a. K-Best
b. Percentile
c. Recursive Feature elimination
6. Transforms
a. PCA
b. ICA

#### Algorithms
Pick an algorithm
Labeled data? (Yes - Supervised, No - Unsupervised)
1. Unsupervised
a. K-means clustering
b. Special clustering
c. PCA
d. Mixture modles / EM algorithm
e. Outlier detection
2. Supervised
a. Ordered or continuous output
a1. Linear Regression
a2. Lasso Regression
a3. Decision Tree Regression
a4. SV regression
b. Non-ordered or discrete ouput
b1. Decision Tree
b2. Naive Bayes
b3. SVM
b4. Ensembles
b5. K nearest neighbors
b6. LDA
b7. Logistic Regression
3. Tune your algorithm
a. parameters of the algorithm
b. Visual inspection
c. Performance on test data
d. GridSearchCV

#### Evaluation
1. Validate
a. Train.Test Split
b. k-fold
c. visualize
2. pick-metrics
a. SSE/r^2
b. Precision
c. recall
d. F1 score
e. ROC curve
f. Custom
g. bias/variance

