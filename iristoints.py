#Shane Healy 09-APR-2018
#Open Iris data set, creates lists and replaces 
#Iris species names as strings with integers.
#Replacing strings with integers makes data more consise. 


import matplotlib.pyplot as plt
import numpy as np 

SL = []#list for Sepal Length
SW = []#list for Sepal Width
PL = []#list for Petal Length
PW = []#list for Petal Width
SPEC = [] #list the Species


with open('data/iris.csv') as f:
    for line in f:
        x = line.split(",")
        SL.append(float(x[0]))
        SW.append(float(x[1]))
        PL.append(float(x[2]))
        PW.append(float(x[3]))
        SPEC.append(str(x[4]))

#replacing strings with integers
for i in range(len(SPEC)):
    if SPEC[i] == "Iris-setosa\n":
        SPEC[i] = 0 
    elif SPEC[i] == "Iris-versicolor\n":
        SPEC[i] = 1   
    elif SPEC[i] == "Iris-virginica\n":
        SPEC[i] = 2 
#printing data set with Iris species represented as integers
for i in range (len(SPEC)):
    print(SL[i],SW[i],PL[i],PW[i],SPEC[i])



