# Programming and Scripting Project APR2018 - Iris Data Set

## Background
Fishers Iris data set<sup>[1](#myfootnote1)</sup> describes the features from fifty samples from each of three species of Iris, *Iris setosa*, *Iris virginica* and *Iris versicolor*. The features, measured in cm are:
*	Sepal Length
*	Sepal width
*	Petal length
*	Petal width  

The Iris data set is real data  
|Sepal Length(cm)|Sepal Width(cm)|Petal Length(cm)|Petal Width(cm)|Class|
|:--------------:|:-------------:|:--------------:|:-------------:|:------------:|
|5.1|3.5|1.4|0.2|Iris-setosa|
2 4.9 3.0 1.4 0.2 Iris-setosa 
3 4.7 3.2 1.3 0.2 Iris-setosa


Sir Ronald Fisher was a British statistician and geneticist. He helped create the foundation for modern statistical science and in 1936 he introduced the Iris flower data set as an example of discriminant analysis<sup>[2](#myfootnote2)</sup>.  
Fisher used linear discriminant analysis to classify Iris species based on four features.  
Linear discriminant analysis is a method used in statistics, pattern recognition and machine learning. The function is to find a linear combination of features that characterizes or separates two or more classes of objects. The resulting combination may be used to classify the objects<sup>[3](#myfootnote3)</sup>.


Irisscatter.py imports Iris data set and splits the data of features in lists. These lists are shown in a scatter plot. 
<br>
<img height="500" src=https://github.com/healyshane/Project_IRIS/blob/master/Iris%20scatter%20plot.png/>
<br>


Iristoints.py imports Iris data set, splits the data in lists and converts the Iris species names from strings to integers. 
This results in a more consise data set that is easier to analyse. 
The conversion is Iris-setosa  = o, Iris-versicolor = 1 and Iris-virginica = 2.




## References
<a name="myfootnote1">1</a>: Wikipedia, Iris flower data set , https://en.wikipedia.org/wiki/Iris_flower_data_set  
<a name="myfootnote2">2</a>: Wikipedia, Ronald Fisher, https://en.wikipedia.org/wiki/Ronald_Fisher  
<a name="myfootnote3">3</a>: Wikipedia, Linear discriminant analysis,  https://en.wikipedia.org/wiki/Linear_discriminant_analysis

