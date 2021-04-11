# Author: Olga Kreicberga 
# A dataset represents a set of 150 records with five attributes in the following order: 
# sepal length, sepal width, petal length, petal width, and a class corresponding to one of three types: 
# Iris Setosa, Iris Versicolor or Iris Virginica, designated 0, 1, 2, respectively.
# Problems needs to be solved: 
# 1. To invistigate code,  
# 2. outputs a summary of each variable to a single text file, 
# 3. saves a histogram of each variable to png files, and 
# 4. outputs a scatter plot of each pair of variables.
# 
# References: 
# http://www.100byte.ru/python/iris/iris.html 
# https://towardsdatascience.comdata-visualization-with-python-8bc988e44f22
# 
#
# Import Librarys (all for the whole project)


import numpy as np 
import matplotlib.pyplot as plt # library is most commonly used in Python in the field of machine learning. 
# It helps in plotting the graph of large dataset.
import seaborn as sns  # provides a beautiful with different 
# styled graph plotting that make dataset more distinguishable and attractive.
import pandas as pd # for data analysis (data manipulation, time-series analysis, 
# integrating indexing of data, etc.).
from sklearn import datasets # to load and process a dataset

#Downloading dataset

iris = pd.read_csv('iris.csv') 

# outputs a summary of each variable to a single text file
# use a 'with' statement that simplifies closure and cleanup tasks.
# In this case, the "close" statement is unnecessary, because "with" will automatically close the file.

with open ('iris_output.txt', 'w') as file:
     file.write(str(iris))

# print a number of elements 
print (list(iris))

# Analyse of dataset
# Methods available for the dataset

print(iris.shape) # (150, 5)
#prints the first 20 lines of dataset
print(iris.head(20)) 
 # count, mean, std, etc. 
print(iris.describe())
# Setosa 50, Versicolor 50, Virginica 50
print(iris.groupby('variety').size()) 

# The histograms of the distribution of attributes Iris dataset
iris['petalLength'].plot.hist()
iris.plot.hist(subplots=True, layout=(2,2), figsize=(8,8 ))
plt.savefig("iris_histograms.png")
plt.show()


# Diagram

# Scatter plots can be used to analyze how one variable affects the other variables.
# By using the for loop we create a single scatter plot of three different species
#  (each represented by a different color).
# I change the colors of dots to BlueViolet, Deeppink and Aquamarine. 
# By using the "for loop" I create a single scatter plot of three different species 
# (each represented by a different color).
colours = {'Setosa':'BlueViolet', 'Versicolor':'DeepPink', 'Virginica':'aquamarine'}
for i in range(len(iris['sepalLength'])):
  plt.scatter(iris['petalLength'][i],iris['petalWidth'][i], color = colours[iris['variety'][i]])
plt.title('Iris')
plt.xlabel('petal Length')
plt.ylabel('petal Width')
plt.grid(True) # enable grid display 

plt.savefig('iris_plotting.png') # save scatter plot 
plt.show()





