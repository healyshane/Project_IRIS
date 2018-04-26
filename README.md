# Programming and Scripting Project APR2018 - Iris Data Set

## Background
Fishers Iris data set<sup>[1](#myfootnote1)</sup> describes the features from fifty samples from each of three classes of Iris, *Iris setosa*, *Iris virginica* and *Iris versicolor*. The features, measured in cm are:
*	Sepal Length
*	Sepal width
*	Petal length
*	Petal width  

The [Iris data set](https://github.com/healyshane/Project_IRIS/blob/master/DATA/iris.csv)<sup>[2](#myfootnote2)</sup> is a real data set, an example of the features for each of the three classes is shown in the table below.

Sepal Length(cm) | Sepal Width(cm) | Petal Length(cm) | Petal Width(cm) | Class 
:--------------: | :-------------: | :--------------: | :-------------: | :-----------------:
5.1 | 3.5 | 1.4 | 0.2 | *Iris-setosa*
7 |	3.2 |	4.7	| 1.4	| *Iris-versicolor*	
6.3 | 3.3 |	6 |	2.5 |	*Iris-virginica*



Sir Ronald Fisher was a British statistician and geneticist. He helped create the foundation for modern statistical science and in 1936 he introduced the Iris flower data set as an example of discriminant analysis<sup>[3](#myfootnote3)</sup>.  
Fisher used linear discriminant analysis to classify Iris class based on four features.  

Linear discriminant analysis is a method used in statistics, pattern recognition and machine learning. The function is to find a linear combination of features that characterizes or separates two or more classes of objects. The resulting combination may be used to classify the objects<sup>[4](#myfootnote4)</sup>. Linear discriminant analysis focuses on maximizing the seperatibility among known categories. This is achieved by projecting the data onto a new axis / plane and maximizing the distance between category means and minimizing the variation within each category<sup>[5](#myfootnote5)</sup>.  

[Irisscatter.py](https://github.com/healyshane/Project_IRIS/blob/master/irisscatter.py) imports the Iris data set and splits the data of features in lists. These lists are shown in a scatter plot. Overlapping of features for the different classes is evident.  

<br>
<img height="500" src=https://github.com/healyshane/Project_IRIS/blob/master/Graphs/Scatter%20Plot%20of%20Iris%20Species.png/>
<br>

<img src="https://github.com/healyshane/Project_IRIS/blob/master/Graphs/Scatter%20plot%20of%20Petal%20Length%20V%20Petal%20Width.png" width="425"/> <img src="https://github.com/healyshane/Project_IRIS/blob/master/Graphs/Scatter%20plot%20of%20Sepal%20Length%20V%20Sepal%20Width.png" width="425"/>



histogram.py
<p align="center">
  <img src="https://github.com/healyshane/Project_IRIS/blob/master/Graphs/Histogram%20of%20Petal%20Length.png" width="525px" height="350px"/></p>
<p align="center">
  <img src="https://github.com/healyshane/Project_IRIS/blob/master/Graphs/Histogram%20of%20Petal%20Width.png" width="525px" height="350px"/></p>
<p align="center">
  <img src="https://github.com/healyshane/Project_IRIS/blob/master/Graphs/Histogram%20of%20Sepal%20Length.png" width="525px" height="350px"/></p>
<p align="center">
  <img src="https://github.com/healyshane/Project_IRIS/blob/master/Graphs/Histogram%20of%20Sepal%20Width.png" width="525px" height="350px"/></p>  

iris_stats_py imports Iris data set, splits the data in lists and converts the Iris species names from strings to integers. 
This results in a more consise data set that is easier to analyse. 
The conversion is *Iris-setosa*  = o, *Iris-versicolor* = 1 and *Iris-virginica* = 2.
X
X  

Parameter                           | *Iris setosa* | *Iris versicolor* | *Iris virginica* | Iris Data Set* 
:---------------------------------- | :-----------: | :---------------: | :--------------: | :---------------:
Min value of Sepal Length | 4.3 | 4.9 | 4.9 | 4.3
Max value of Sepal Length | 5.8| 7 | 7.9 | 7.9
Mean value of Sepal Length |5.006 | 5.936 | 6.588 | 5.843333333
Standard Deviation of Sepal Length | 0.348946987 | 0.510983366 | 0.629488681 | 0.825301292
 |   |   |   |  
 |   |   |   |  
Min value of Sepal Width | 2.3 | 2 | | 2.2 | 2
Max value of Sepal Width | 4.4 | 3.4 | 3.8 | 4.4
Mean value of Sepal Width | 3.418 | 2.77 | 2.974 | 3.054
Standard Deviation of Sepal Width | 0.37719491 | 0.310644491 | 0.319255384 | 0.43214658 
 |   |   |   |   
 |   |   |   |  
Min value of Petal Length | 1 | 3 | 4.5 | 1
Max value of Petal Length | 1.9 | 5.1 | 6.9 | 6.9
Mean value of Petal Length | 1.464 | 4.26 | 5.552 | 3.758666667
Standard Deviation of Petal Length | 0.171767284 | 0.465188134 | 0.546347875 | 1.758529183
 |   |   |   |
 |   |   |   |  
Min value of Petal Width | 0.1 | 1 | 1.4 | 0.1
Max value of Petal Width | 0.6 | 1.8 | 2.5 | 2.5
Mean value of Petal Width | 0.244 | 1.326 | 2.026 | 1.198666667
Standard Deviation of Petal Width | 0.106131993 | 0.195765165 | 0.271889684 | 0.760612619


![Comparison between iris means](https://github.com/healyshane/Project_IRIS/blob/master/Graphs/Mean%20Comparison.png)  


# Machine Learning
Machine learning uses statistics to give computer systems the ability to learn without being explicitly programmed. Training data is required to devise algorithms or models that will predict an output or find patterns in the data. Machine learning is applicable to a broad range of subjects, from medical diagnosis to financial market speculation to product marketing. Machine learning may be supervised or unsupervised.   

## Supervised and Unsupervised Machine Learning Algorithms<sup>[3](#myfootnote3)</sup>  
Supervised learning has input variables and an output variable. An algorithm is used to learn the mapping function from the input to the output. Objective with supervised learning is to be able to correctly predict the output given particular inputs. Within supervised learning, a classification problem would be where the output is a category while a regression problem would be where the output is a real value. 
Unsupervised learning has input data but no corresponding outsput variables. An algorithm is used to model the structure of the data. Within unsupervised leaning, a clustering problem would be where groupings in the data is required while an association problem would be defining rules that govern the data. 
Semi-Supervised Machine Learning is used for problems with input data and only some of the data is labelled. Many real world machine learning problems fall into this area and a mixture of supervised and unsupervised learning methods will be used to analyse data.  

Depending on the desired outcome of machine learning, different techniques may be applied<sup>[3](#myfootnote3)</sup>. 
* Classification - Classifiers act as functions where the training data is used to adjust the parameters of the function model. The quality of the data is important in machine learning classification. Independent / unbiased and distinct features are required as inputs to promote accuracy.
* Regression - Estimating the relationships between variables.
* Clustering - Inputs are divided into groups that are unknown.
* Denstiy Estimation - Determines the distribution of inputs
* Dimensionality reduction - Simplifies inputs by mapping them to a lower-dimensional space. Linear discriminant analysis, as used by Fisher to classify Iris class, is an example of dimensionality reduction. 

## Scikit-learn Project

The scikit-learn project<sup>[3](#myfootnote3)</sup> provides an open source machine learning library for the Python programming language. The library is a collection of classes and functions that are imported into Python programs and centred around the NumPy and SciPy libraries. All objects within scikit-learn share three complementary interfaces:  
1. Estimator - Develops a fit method for learning a model from training data. 
1. Predictor – Uses an array to produce predictions based on the learned parameters of the estimator and scores the accuracy.
1. Transformer – To allow filtering and modification of data before feeding into learning algorithm.  

## Decision Tree in Machine Learning
In YouTube video, Visualizing a Decision Tree – Machine Learning Recipes #2<sup>[8](#myfootnote8)</sup>, a decision tree is used as a predictive model to map observations about the data and to visualise how the classifier works. The goals are to import the Iris dataset, train a classifier, predict label for a new flower and visualize the decision tree. 

To begin numpy and sklearn packages and Iris data set are imported into Python. test_idx identifies one example of each type of flower. 

```python
import numpy as np 
from sklearn.datasets import load_iris
from sklearn import tree

iris = load_iris()
test_idx = [0,50,100] 
```
One example of each type of flower are removed from data and target variables, this provides values for testing to check accuracy of classifier. 
Two new sets of variables are created, one for training and one for testing.

```python
train_target = np.delete(iris.target, test_idx)
train_data = np.delete(iris.data, test_idx, axis = 0)

test_target = iris.target[test_idx]
test_data = iris.data[test_idx]
```
A decision tree classifier is created and trained on training data. 
```python
clf = tree.DecisionTreeClassifier()
clf.fit(train_data, train_target)
```
The labels of testing data matches the predicted labels and are printed to screen.
This demonstrates that, based on the testing data, the classifier was successful. 
```python
print(test_target) 
print(clf.predict(test_data))
```
To visualise how the classifier works, the decision tree<sup>[9](#myfootnote8)</sup> is exported in Graphviz format.
<p align="center">
  <img src="https://github.com/healyshane/Project_IRIS/blob/master/Graphs/Iris%20Classifier%20Decision%20Tree.png" width="750px" height="500px"/></p>  
Sepal Width is not referenced in the decision tree. XXXX
  
## References
<a name="myfootnote1">1</a>: Wikipedia, Iris flower data set , https://en.wikipedia.org/wiki/Iris_flower_data_set  
<a name="myfootnote2">2</a>: UCI, Machine Learning Repository, http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data  
<a name="myfootnote3">3</a>: Wikipedia, Ronald Fisher, https://en.wikipedia.org/wiki/Ronald_Fisher  
<a name="myfootnote4">4</a>: Wikipedia, Linear discriminant analysis,  https://en.wikipedia.org/wiki/Linear_discriminant_analysis  
<a name="myfootnote5">5</a>: StatQuest: Linear Discriminant Analysis (LDA) clearly explained,  https://www.youtube.com/watch?v=azXCzI57Yfc  
<a name="myfootnote6">6</a>: Supervised and Unsupervised Machine Learning Algorithm, https://machinelearningmastery.com/supervised-and-unsupervised-machine-learning-algorithms/  
<a name="myfootnote7">7</a>: API design for machine learning software: experiences from the scikit-learn project, https://arxiv.org/abs/1309.0238 )  
<a name="myfootnote8">8</a>: Google Developers, YouTube Playlist Machine Learning Recipes with Josh Gordon https://www.youtube.com/playlist?list=PLOU2XLYxmsIIuiBfYad6rFYQU_jL2ryal  
<a name="myfootnote8">8</a>: scikit-learn Documentation, http://scikit-learn.org/stable/modules/tree.html  
<a name="myfootnote1">1</a>: Wikipedia, Machine Learning, https://en.wikipedia.org/wiki/Machine_learning 




