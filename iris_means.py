# Shane Healy, 24-APR-2018
# Code to import Iris_stats.csv which contains statistical data on Iris data set. 
# Identify data that represent the mean values of Iris features. 
# Plot bar charts to illustrate difference in means for features. 

import numpy as np 
import matplotlib.pyplot as plt

# Generate an array from Iris Data Set.
data = np.genfromtxt('data/iris_stats.csv',delimiter = ',')

# Create variables identified from Iris_stats.csv with:
# MSL = Mean Sepal Length
# MSW = Mean Sepal Width
# MPL = Mean Petal Width
# MPW = Mean Petal Width
MSL = data[3,1:]
MSW = data[7,1:]
MPL = data[11,1:]
MPW = data[15,1:]

#y_pos is the position of the bars and /
#np.arange creates evenly spaced ticks for items within Species.
Species = ('Iris setosa','Iris versicolor','Iris virginica','Iris data')
y_pos = np.arange(len(Species))

#Defining fig to allow for subplots. 
fig = plt.figure()

#Create bar subplot for Mean Sepal Lengths. 
plt.subplot(2, 2, 1)
plt.bar(y_pos, MSL, align='center', alpha=0.5, color='red')
plt.xticks(y_pos, Species)
plt.ylabel('cm', fontsize=12)
plt.title('Mean Sepal lengths', fontsize=14)

#Create bar subplot for Mean Sepal Widths. 
plt.subplot(2, 2, 2)
plt.bar(y_pos, MSW, align='center', alpha=0.5, color='green')
plt.xticks(y_pos, Species)
plt.ylabel('cm', fontsize=12)
plt.title('Mean Sepal Widths', fontsize=14)

#Create bar subplot for Mean Petal Lengths. 
plt.subplot(2, 2, 3)
plt.bar(y_pos, MPL, align='center', alpha=0.5, color='blue')
plt.xticks(y_pos, Species)
plt.ylabel('cm', fontsize=12)
plt.title('Mean Petal Lengths', fontsize=14)

#Create bar subplot for Mean Petal Widths. 
plt.subplot(2, 2, 4)
plt.bar(y_pos, MPW, align='center', alpha=0.5, color='black')
plt.xticks(y_pos, Species)
plt.ylabel('cm', fontsize=12)
plt.title('Mean Petal Widths', fontsize=14)

plt.show()
