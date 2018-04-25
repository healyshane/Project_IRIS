#Shane Healy, 10-APR-2018, inserted comments in/
# code reproduced from YouTube video https://www.youtube.com/watch?v=tNa99PG8hR8&index=2&list=PLOU2XLYxmsIIuiBfYad6rFYQU_jL2ryal&t=251s 
# In analysis, 0 = Iris setosa, 1 = Iris versicolor, 2 = Iris virginica
import numpy as np 
from sklearn.datasets import load_iris
from sklearn import tree

#loads Iris data set
iris = load_iris()
#Identifies one example of data for each of the Iris species, for testing. 
test_idx = [0,50,100]

#Removing the testing data to generate training data for model. 
train_target = np.delete(iris.target, test_idx)
train_data = np.delete(iris.data, test_idx, axis = 0)

#Creating variables for testing data
test_target = iris.target[test_idx]
test_data = iris.data[test_idx]

#Defines the classifier and fits the model to the training data.
clf = tree.DecisionTreeClassifier()
clf.fit(train_data, train_target)

print(test_target) # prints the known results of testing data. 
print(clf.predict(test_data)) # prints the preticted results of testing data.

#Prints the feature names and target names(species)
print(iris.feature_names, iris.target_names)

#prints the features and species of each of the testing data. 
print(test_data[0],test_target[0])
print(test_data[1],test_target[1])
print(test_data[2],test_target[2])

#features for 4 ficticious examples created. 
W = [6.2, 2.8, 5.1, 1.5]
X = [6, 2, 2.3, 1.2]
Y = [6.8, 2.8, 5.0, 1.57]
Z = [7, 2.8, 5.0, 1.8]

#Print the predicted species considering the features inputted. 
print(clf.predict([W]))
print(clf.predict([X]))
print(clf.predict([Y]))
print(clf.predict([Z]))

