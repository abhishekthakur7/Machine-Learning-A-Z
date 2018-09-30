#Simple linear regression method
# Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Salary_Data.csv')
#dataset.iloc[ , ] --> iloc[ <rows either slice/list or single value> , <columns either slice/list or single value> ]
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)

#Feature scaling - LinearRegressionModel will handle it automatically

#Fitting data to simple linear regression model
from sklearn.linear_model import LinearRegression
regressor= LinearRegression()
regressor.fit(X_train, y_train) #this will create a unique line equation, based on train data 


#predicting the test set results
y_prid = regressor.predict(X_test)

#Visualizing training set results
plt.scatter(X_train, y_train, color='red')
plt.plot(X_train, regressor.predict(X_train), color='blue') #Compare real salaries and predicted salaries
plt.xlabel('Experience')
plt.ylabel('Salary($)')
plt.title('Experience vs Salary($) (Training Set)')
plt.show()


#Visualizing test set results
plt.scatter(X_test, y_test, color='green')

#doen't matter if you use X_train or X_test as line equation being used to predict will remain the same for "regressor"
plt.plot(X_train, regressor.predict(X_train), color='blue') 
plt.xlabel('Experience')
plt.ylabel('Salary($)')
plt.title('Experience vs Salary($) (Test Set)')
plt.show()