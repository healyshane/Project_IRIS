# Programming and Scripting Project APR2018 - Iris Data Set

## Background
Fishers Iris data set<sup>[1](#myfootnote1)</sup> describes the features from fifty samples from each of three classes of Iris, *Iris setosa*, *Iris virginica* and *Iris versicolor*. The features, measured in cm are:
*	Sepal Length
*	Sepal width
*	Petal length
*	Petal width  

The [Iris data set](https://github.com/healyshane/Project_IRIS/blob/master/iris.csv)<sup>[2](#myfootnote2)</sup> is a real data set, an example of the features for each of the three classes is shown in the table below.

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
<img height="500" src=https://github.com/healyshane/Project_IRIS/blob/master/Graphs/Iris%20scatter%20plot.png/>
<br>

histogram.py
<p align="center">
  <img src="https://github.com/healyshane/Project_IRIS/blob/master/Graphs/Histogram%20of%20Petal%20Length.png" width="525px" height="350px"/></p>
<p align="center">
  <img src="https://github.com/healyshane/Project_IRIS/blob/master/Graphs/Histogram%20of%20Petal%20Width.png" width="525px" height="350px"/></p>
<p align="center">
  <img src="https://github.com/healyshane/Project_IRIS/blob/master/Graphs/Histogram%20of%20Sepal%20Length.png" width="525px" height="350px"/></p>
<p align="center">
  <img src="https://github.com/healyshane/Project_IRIS/blob/master/Graphs/Histogram%20of%20Sepal%20Width.png" width="525px" height="350px"/></p>  

Iristoints.py imports Iris data set, splits the data in lists and converts the Iris species names from strings to integers. 
This results in a more consise data set that is easier to analyse. 
The conversion is *Iris-setosa*  = o, *Iris-versicolor* = 1 and *Iris-virginica* = 2.

## Decision Tree in Machine Learning
Google Developers, YouTube Playlist Machine Learning Recipes with Josh Gordon https://www.youtube.com/playlist?list=PLOU2XLYxmsIIuiBfYad6rFYQU_jL2ryal 
In YouTube video, Visualizing a Decision Tree – Machine Learning Recipes #2 REF, a decision tree is used to visualise how the classifier works. The goals are to import dataset, train a classifier, predict label for new flower and visualize the decision tree. 

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
To visualise how the classifier works, the decision treeREFSCI-LEARN is exported in Graphviz format.
<p align="center">
  <img src="https://github.com/healyshane/Project_IRIS/blob/master/Graphs/Iris%20Classifier%20Decision%20Tree.png" width="750px" height="500px"/></p> 
## References
<a name="myfootnote1">1</a>: Wikipedia, Iris flower data set , https://en.wikipedia.org/wiki/Iris_flower_data_set  
<a name="myfootnote2">2</a>: UCI, Machine Learning Repository, http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data  
<a name="myfootnote3">3</a>: Wikipedia, Ronald Fisher, https://en.wikipedia.org/wiki/Ronald_Fisher  
<a name="myfootnote4">4</a>: Wikipedia, Linear discriminant analysis,  https://en.wikipedia.org/wiki/Linear_discriminant_analysis  
<a name="myfootnote5">5</a>: StatQuest: Linear Discriminant Analysis (LDA) clearly explained,  https://www.youtube.com/watch?v=azXCzI57Yfc  



