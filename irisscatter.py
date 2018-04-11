#Shane Healy 04-APR-2018
#Open Iris data set and create Scatter Plot of features. 

import matplotlib.pyplot as plt
import numpy as np 

SL = []#create empty list for Sepal Length
SW = []#create empty list for Sepal Width
PL = []#create empty list for Petal Length
PW = []#create empty list for Petal Width
SPEC = []#create empty list for Iris Species

#split data into lists
with open('data/iris.csv') as f:
    for line in f:
        x = line.split(",")
        SL.append(float(x[0]))
        SW.append(float(x[1]))
        PL.append(float(x[2]))
        PW.append(float(x[3]))
        SPEC.append(str(x[4]))
        
#scatter plot of data with title, legend and x-axis label
plt.scatter(SL,SPEC,label='Sepal Length',color='purple')
plt.scatter(SW,SPEC,label='Sepal Width')
plt.scatter(PL,SPEC,label='Petal Length')
plt.scatter(PW,SPEC,label='Petal Width')
plt.legend(loc='upper left')
plt.xlabel('cm',fontsize=18,color='purple')
plt.title('Scatter Plot of Iris Flowers', color = 'purple', fontsize=18)
plt.show()

