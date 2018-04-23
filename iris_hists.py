# Shane Healy 22-APR-2018
# Goal is to import Iris data set and create histograms for each feature.
# The histograms display distribution of features for Iris setosa, Iris versicolor and Iris virginica.

import numpy as np 
import matplotlib.pyplot as plt 

# Generate an array from Iris Data Set.
data = np.genfromtxt('data/iris.csv',delimiter = ',')


# Identifying variables to hold the first four columns of data set. 
Sepal_Length = data[:,0]
Sepal_Width = data[:,1]
Petal_Length = data[:,2]
Petal_Width = data[:,3]

# Histogram for displaying Sepal Length. 
plt.hist(Sepal_Length,color='skyblue',edgecolor='black')
plt.title('Histogram of Sepal Length',size=18)
plt.xlabel('Sepal Length (cm)', size=14)
plt.ylabel('Frequency', size=14)
plt.grid(True)
plt.show()

# Histogram for displaying Sepal Width. 
plt.hist(Sepal_Width,color='grey',edgecolor='black')
plt.title('Histogram of Sepal Width',size=18)
plt.xlabel('Sepal Width (cm)', size=14)
plt.ylabel('Frequency', size=14)
plt.grid(True)
plt.show()

# Histogram for displaying Petal Length. 
plt.hist(Petal_Length,color='fuchsia',edgecolor='black')
plt.title('Histogram of Petal Length',size=18)
plt.xlabel('Petal Length (cm)', size=14)
plt.ylabel('Frequency', size=14)
plt.grid(True)
plt.show()

# Histogram for displaying Petal Width. 
plt.hist(Petal_Width,color='darkcyan',edgecolor='black')
plt.title('Histogram of Petal Width',size=18)
plt.xlabel('Petal Width (cm)', size=14)
plt.ylabel('Frequency', size=14)
plt.grid(True)
plt.show()