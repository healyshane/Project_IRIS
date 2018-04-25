#Shane Healy 09-APR-2018, 20-APR-2018
#Open Iris data set, creates lists and replaces 
#Iris species names as strings with integers.
#Replacing strings with integers makes data more consise. 

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

#replacing strings of Iris species with integers
for i in range(len(SPEC)):
    if SPEC[i] == "Iris-setosa\n":
        SPEC[i] = 0 
    elif SPEC[i] == "Iris-versicolor\n":
        SPEC[i] = 1   
    elif SPEC[i] == "Iris-virginica\n":
        SPEC[i] = 2 

# Define a function to loop over data set and create an array holding data on which analysis can be performed.  
def Iris(species,feature):
    results = []
    for i in range (len(SPEC)):
        if SPEC[i] == species:
            results.append(float(feature[i]))
            p = np.array([results])
    return p

# Allow a user to input a species selection to print statistical information. 
# 0 for Iris-setosa, 1 for Iris-versicolor, 2 for Iris-virginica, 3 for all species.
# If the feature could also be entered by user, it would severly reduce the required number of print statements in following code. 

Species = int(input("""Insert number for statistical information on: 
    0 : Iris Setosa
    1 : Versicolor
    2 : Virginica
    3 : All species in Iris data set
and press Enter. 
"""))
print("")


# Feature is SL for Sepal Length, SW for Sepal Width, PL for Petal Length and PW for Petal Width. 

# If "All species in Iris data set" is selected /
# Min / Max / Mean and standard deviation of entire data set is printed for each of the features. 
if Species == 3:
    print("Statistical information for all species in Iris data:")
    print("")
    print('Min value of Sepal Length in data set is', np.amin(SL))
    print('Max value of Sepal Length in data set is', np.amax(SL))
    print('Mean value of Sepal Length in data set is', np.mean(SL, dtype=float))
    print('Standard Deviation of Sepal Length in data set is', np.std(SL, dtype=float))
    print("")
    
    print('Min value of Sepal Width in data set is', np.amin(SW))
    print('Max value of Sepal Width in data set is', np.amax(SW))
    print('Mean value of Sepal Width in data set is', np.mean(SW, dtype=float))
    print('Standard Deviation of Sepal Width in data set is', np.std(SW, dtype=float))
    print("")
    
    print('Min value of Petal Length in data set is', np.amin(PL))
    print('Max value of Petal Length in data set is', np.amax(PL))
    print('Mean value of Petal Length in data set is', np.mean(PL, dtype=float))
    print('Standard Deviation of Petal Length in data set is', np.std(PL, dtype=float))
    print("")
    
    print('Min value of Petal Width in data set is', np.amin(PW))
    print('Max value of Petal Width in data set is', np.amax(PW))
    print('Mean value of Petal Width in data set is', np.mean(PW, dtype=float))
    print('Standard Deviation of Petal Width in data set is', np.std(PW, dtype=float))
    print("")
    exit() 
    
# Printing header to describe output. 
elif Species == 0:
    print("Statistical information for Iris setosa:")
elif Species == 1:
    print("Statistical information for Iris versicolor:")
else: 
    print("Statistical information for Iris virginica:")
    
print("")

#Prints the statistical information on requested species of Iris by calling function Iris. 

print(f'Min value of Sepal Length is {np.amin(Iris(Species,SL))}.')
print(f'Max value of Sepal Length is {np.amax(Iris(Species,SL))}.')
print(f'Mean value of Sepal Length is {np.mean(Iris(Species,SL),dtype=float)}.')
print(f'Standard Deviation of Sepal Length is {np.std(Iris(Species,SL),dtype=float)}.')
print("")

print(f'Min value of Sepal Width is {np.amin(Iris(Species,SW))}.')
print(f'Max value of Sepal Width is {np.amax(Iris(Species,SW))}.')
print(f'Mean value of Sepal Width is {np.mean(Iris(Species,SW),dtype=float)}.')
print(f'Standard Deviation of Sepal Width is {np.std(Iris(Species,SW),dtype=float)}.')
print("")

print(f'Min value of Petal Length is {np.amin(Iris(Species,PL))}.')
print(f'Max value of Petal Length is {np.amax(Iris(Species,PL))}.')
print(f'Mean value of Petal Length is {np.mean(Iris(Species,PL),dtype=float)}.')
print(f'Standard Deviation of Petal Length is {np.std(Iris(Species,PL),dtype=float)}.')
print("")

print(f'Min value of Petal Width is {np.amin(Iris(Species,PW))}.')
print(f'Max value of Petal Width is {np.amax(Iris(Species,PW))}.')
print(f'Mean value of Petal Width is {np.mean(Iris(Species,PW),dtype=float)}.')
print(f'Standard Deviation of Petal Width is {np.std(Iris(Species,PW),dtype=float)}.')
