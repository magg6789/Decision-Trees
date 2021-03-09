#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Import Python Libraries: NumPy and Pandas
import pandas as pd
import numpy as np


# In[2]:


# Import Libraries & modules for data visualization
from pandas.plotting import scatter_matrix
from matplotlib import pyplot


# In[3]:


# Import scikit-Learn module for the algorithm/modeL: Nearest Neighbors
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeRegressor


# In[4]:


# Import scikit-Learn module to split the dataset into train/ test sub-datasets
from sklearn.model_selection import train_test_split


# In[5]:


# Import scikit-Learn module for K-fold cross-validation - algorithm/modeL evaluation & validation
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score


# In[6]:


# Import scikit-Learn module classification report to later use for information about how the system try to classify / lable each record
from sklearn.metrics import classification_report


# # Load the data

# In[7]:


#Specify location of the dataset
filename = 'C:/Users/miriamgarcia/Downloads/housing_boston_w_hdrs.csv'
df = pd.read_csv(filename)
print(df)


# # Preprocess Dataset

# In[8]:


df = df.drop("ZN",1)
df = df.drop("CHAS",1)


# In[9]:


# count the number of NaN values in each
print(df.isnull().sum())


# # Perform the exploratory data analysis (EDA) on the dataset

# In[10]:


# get the dimensions or shape of the dataset
# i.e. number of records / rows X number of variables / columns
print(df.shape)


# In[11]:


#get the data types of all the variables / attributes in the data set
print(df.dtypes)


# In[12]:


#return the first five records / rows of the data set
print(df.head(5))


# In[13]:


#return the summary statistics of the numeric variables / attributes in the data set
print(df.describe())


# In[14]:


#class distribution i.e. how many records are in each class
print(df.groupby('MEDV').size())


# In[15]:


df.hist(figsize=(12, 8))
pyplot.show()


# In[16]:


# generate density plots of each numeric variable / attribute in the data set
df.plot(kind='density', subplots=True, layout=(12, 2), sharex=False, legend=True, fontsize=1,
figsize=(12, 16))
pyplot.show()


# In[17]:


# generate box plots of each numeric variable / attribute in the data set
df.plot(kind='box', subplots=True, layout=(12,2), sharex=False, figsize=(12,8))
pyplot.show()


# In[57]:


# generate scatter plot matrix of each numeric variable / attribute in the data set
scatter_matrix(df, alpha=0.8, figsize=(15, 15))
pyplot.show()


# # Separate Dataset into Input & Output NumPy arrays

# In[19]:


# store dataframe values into a numpy array
array = df.values
# separate array into input and output by slicing
# for X(input) [:, 1:5] --> all the rows, columns from 1 - 4 (5 - 1)
# these are the independent variables or predictors
X = array[:,0:11]
# for Y(input) [:, 5] --> all the rows, column 5
# this is the value we are trying to predict
Y = array[:,11]


# # Split Input/Output Arrays into Training/Testing Datasets

# In[20]:


# split the dataset --> training sub-dataset: 67%; test sub-dataset: 33%
test_size = 0.33
#selection of records to include in each data sub-dataset must be done randomly
seed = 4
#split the dataset (input and output) into training / test datasets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size,
random_state=seed)
print(Y_train)


# # Build and Train the Model

# In[21]:


#build the model
model = DecisionTreeRegressor()
# train the model using the training sub-dataset
model.fit(X_train, Y_train)


# # Prediction 1

# In[22]:


# The suburb area has the following predictors:
#CRIM:1.32
#INDUS :10.34
#NOX:0.52
#RM: 6.0
#AGE: proportion of owner-occupied units built prior to 1940 = 60
#DIS: weighted distances to five Boston employment centers = 3.5
#RAD: index of accessibility to radial highways = 5.0
#TAX: 275    
#PTRATIO: pupil-teacher ratio by town = 16
#B:330
#LSTAT:9.0

model.predict([[1.32, 10.34, 0.52, 6, 60,3.5,5,275,16,330,9]])


# So, the model predict that the median value of owner-occupied homes in 1000 dollars in the above suburb should be around 25,000.

# # Evaluate/Validate Algorithm/Model • Using K-Fold Cross-Validation

# In[23]:


# Evaluate the algorithm
# Specify the K-size
num_folds = 10
# Fix the random seed
# must use the same seed value so that the same subsets can be obtained
# for each time the process is repeated
seed = 4
# Split the whole data set into folds
kfold = KFold(n_splits=num_folds, random_state=seed)
# For Linear regression, we can use MSE (mean squared error) value
# to evaluate the model/algorithm
scoring = 'neg_mean_squared_error'
# Train the model and run K-foLd cross-validation to validate/evaluate the model
results = cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
# Print out the evaluation results
# Result: the average of all the results obtained from the k-foLd cross-validation
print(results.mean())


# After we train we evaluate
# Use K-Fold to determine if the model is acceptable
# We pass the whole set because the system will divide for us
# -42.99 avg of all error (mean of square errors) this value would traditionally be positive value, but scikit reports shows it as negative . Square root of this value is around 6.3 to 6.6

# # Prediction 2

# In[24]:


# The suburb area has the following predictors:
#CRIM:0.6
#INDUS :17.35
#NOX:0.75
#RM: 4.0
#AGE: proportion of owner-occupied units built prior to 1940 = 25
#DIS: weighted distances to five Boston employment centers = 8.5
#RAD: index of accessibility to radial highways = 3.0
#TAX: 400    
#PTRATIO: pupil-teacher ratio by town = 18
#B:340
#LSTAT:12.00

model.predict([[0.6, 17.35, 0.75, 4.0, 25,8.5,3.0,400,18,340,12]])


# So, the model predict that the median value of owner-occupied homes in 1000 dollars in the above suburb should be around 15,300.

# # Evaluate/Validate Algorithm/Model • Using K-Fold Cross-Validation

# In[26]:


# Evaluate the algorithm
# Specify the K-size
num_folds = 10
# Fix the random seed
# must use the same seed value so that the same subsets can be obtained
# for each time the process is repeated
seed = 4
# Split the whole data set into folds
kfold = KFold(n_splits=num_folds, random_state=seed)
# For Linear regression, we can use MSE (mean squared error) value
# to evaluate the model/algorithm
scoring = 'neg_mean_squared_error'
# Train the model and run K-foLd cross-validation to validate/evaluate the model
results = cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
# Print out the evaluation results
# Result: the average of all the results obtained from the k-foLd cross-validation
print(results.mean())


# After we train we evaluate
# Use K-Fold to determine if the model is acceptable
# We pass the whole set because the system will divide for us
# -39.03 avg of all error (mean of square errors) this value would traditionally be positive value, but scikit reports shows it as negative . The Sqaure root of this value would be around 6 to 6.5 (6.240)

# ### Calculate R square value 

# In[27]:


R_squared = model.score(X_test, Y_test)
print(R_squared)


# In[ ]:





# In[ ]:





# In[ ]:




