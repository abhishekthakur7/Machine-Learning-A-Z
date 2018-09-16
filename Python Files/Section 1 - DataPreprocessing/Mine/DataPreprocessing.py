# Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Data.csv')
#dataset.iloc[ , ] --> iloc[ <rows either slice/list or single value> ,  <columns either slice/list or single value> ]
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 3].values

# Taking care of missing data - import Imputer class
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0) #fit mean to NaN value for column index 1 and 2
imputer = imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])

#Encoding catagorical data to numbers -> e.g. Country or Purchased value (Yes/No)
# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 0] = labelencoder_X.fit_transform(X[:, 0])
#create dummy variables for country - as encoded values are not relational i.e. none them is either small or larger than other
onehotencoder = OneHotEncoder(categorical_features = [0])
X = onehotencoder.fit_transform(X).toarray()
# Encoding the Dependent Variable
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
    
#Feature scaling - is required because most machine learning models use euclidean distance
# and salary and age are not scalable w.r.t. each other which means in eclidean distance, salary will dominate age
#because salary values are much larger than age values
#scaling is of two types - 1. Standardisation 2. Normalization
#convert values to +1 -> -1 range
from sklearn.preprocessing import StandardScaler

sc_X = StandardScaler()
X_train= sc_X.fit_transform(X_train)
X_test= sc_X.transform(X_test) #use only transform method on test data

#no scaling for dependent variables because it's categorical data - Yes/No

























