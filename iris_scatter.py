#Shane Healy 04-APR-2018
#Open Iris data set and create Scatter Plot of features. 

import matplotlib.pyplot as plt


SL = []#create empty list for Sepal Length
SW = []#create empty list for Sepal Width
PL = []#create empty list for Petal Length
PW = []#create empty list for Petal Width
SPEC = []#create empty list for Iris Species

#Opens Iris.csv file, each line is split and iteratively appended into defined list 
with open('data/iris.csv') as f:
    for line in f:
        x = line.split(",")
        SL.append(float(x[0]))
        SW.append(float(x[1]))
        PL.append(float(x[2]))
        PW.append(float(x[3]))
        SPEC.append(str(x[4]))
        
#Scatter plot of the four features and Iris species. 
plt.scatter(SL,SPEC, color = "red", label='Sepal Length',marker='h',s=150)
plt.scatter(SW,SPEC, color = "blue", label='Sepal Width',marker='*',s=150)
plt.scatter(PL,SPEC, color = "green", label='Petal Length',marker='D',s=150)
plt.scatter(PW,SPEC, color = "black", label='Petal Width',marker='o',s=150)
plt.legend(loc='upper left',fontsize=16)
plt.xlabel('cm',fontsize=18,color='black')
plt.ylabel('Species',fontsize=18,color='black')
plt.title('Scatter Plot of Iris Species', color = 'black', fontsize=22)
plt.tick_params(axis='both', labelsize=16)
plt.grid(True,axis='n')
plt.show()

#scatter plot of Sepal Length and Sepal Width
plt.scatter(SPEC,SL, color = "red", label='Sepal Length')
plt.scatter(SPEC,SW, color = "blue", label='Sepal Width')
plt.legend(loc='upper left',fontsize=16)
plt.xlabel('Species',fontsize=18,color='black')
plt.ylabel('cm',fontsize=18,color='black')
plt.title('Sepal Length V Sepal Width', color = 'black', fontsize=20)
plt.tick_params(axis='both', labelsize=16)
plt.grid(True,axis='y')
plt.show()

#Scatter plot of Petal Length and Petal Width
plt.scatter(SPEC,PL, color = "green", label='Petal Length')
plt.scatter(SPEC,PW, color = "black", label='Petal Width')
plt.legend(loc='upper left',fontsize=16)
plt.xlabel('Species',fontsize=18,color='black')
plt.ylabel('cm',fontsize=18,color='black')
plt.title('Petal Length V Petal Width', color = 'black', fontsize=20)
plt.tick_params(axis='both', labelsize=16)
plt.grid(True,axis='y')
plt.show()

#Scatter plot of Sepal Length and Petal Length
plt.scatter(SPEC,SL, color = "red", label='Sepal Length')
plt.scatter(SPEC,PL, color = "green", label='Petal Length')
plt.legend(loc='upper left',fontsize=16)
plt.xlabel('Species',fontsize=18,color='black')
plt.ylabel('cm',fontsize=18,color='black')
plt.title('Sepal Length V Petal Length', color = 'black', fontsize=20)
plt.tick_params(axis='both', labelsize=16)
plt.grid(True,axis='y')
plt.show()

#Scatter plot of Sepal Width and Petal Width
plt.scatter(SPEC,SW, color = "blue", label='Sepal Width')
plt.scatter(SPEC,PW, color = "black", label='Petal Width')
plt.legend(loc='upper left',fontsize=16)
plt.xlabel('Species',fontsize=18,color='black')
plt.ylabel('cm',fontsize=18,color='black')
plt.title('Sepal Width V Petal Width', color = 'black', fontsize=20)
plt.tick_params(axis='both', labelsize=16)
plt.grid(True,axis='y')
plt.show()

#Scatter plot of Sepal Width and Petal Length
plt.scatter(SPEC,PL, color = "green", label='Petal Length')
plt.scatter(SPEC,SW, color = "blue", label='Sepal Width')
plt.legend(loc='upper left',fontsize=16)
plt.xlabel('Species',fontsize=18,color='black')
plt.ylabel('cm',fontsize=18,color='black')
plt.title('Petal Length V Sepal Width', color = 'black', fontsize=20)
plt.tick_params(axis='both', labelsize=16)
plt.grid(True,axis='y')
plt.show()

#Scatter plot of Sepal Length and Petal Width
plt.scatter(SPEC,PW, color = "black", label='Petal Width')
plt.scatter(SPEC,SL, color = "red", label='Sepal Length')
plt.legend(loc='upper left',fontsize=16)
plt.xlabel('Species',fontsize=18,color='black')
plt.ylabel('cm',fontsize=18,color='black')
plt.title('Petal Width V Sepal Length', color = 'black', fontsize=20)
plt.tick_params(axis='both', labelsize=16)
plt.grid(True,axis='y')
plt.show()
