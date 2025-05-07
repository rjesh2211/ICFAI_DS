'''


Simple Linear Regression
In this example we will consider sales based on 'TV' marketing budget.

In this notebook, we'll build a linear regression model to predict 'Sales' using 'TV' as the predictor variable.

Understanding the Data
Let's start with the following steps:

Importing data using the pandas library
Understanding the structure of the data
'''

import pandas as pd

advertising = pd.read_csv('tvmarketing.csv')


advertising.head()


advertising.info()

advertising.describe()


advertising.plot(x='TV',y='Sales',kind='scatter')


'''

y=c+m1×TV

The m
 values are called the model coefficients or model parameters.

Generic Steps in Model Building using sklearn
Before you read further, it is good to understand the generic structure of modeling using the scikit-learn library. Broadly, the steps to build any model can be divided as follows:

Preparing X and y
The scikit-learn library expects X (feature variable) and y (response variable) to be NumPy arrays.
However, X can be a dataframe as Pandas is built over NumPy.
'''

# Putting feature variable to X
X = advertising['TV']

# Print the first 5 rows
X.head()



# Putting response variable to y
y = advertising['Sales']

# Print the first 5 rows
y.head()


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7 , random_state=0000)



print(type(X_train))
print(type(X_test))
print(type(y_train))
print(type(y_test))


#It is a general convention in scikit-learn that observations are rows, while features are columns. 
#This is needed only when you are using a single feature; in this case, 'TV'.

import numpy as np
#Simply put, numpy.newaxis is used to increase the dimension of the existing array by one more dimension,
X_train = X_train[:, np.newaxis]
X_test = X_test[:, np.newaxis]


# import LinearRegression from sklearn
from sklearn.linear_model import LinearRegression

# Representing LinearRegression as lr(Creating LinearRegression Object)
lr = LinearRegression()

# Fit the model using lr.fit()
lr.fit(X_train, y_train)


# Print the intercept and coefficients
print(lr.intercept_)
print(lr.coef_)



y_pred = lr.predict(X_test)




from sklearn.metrics import mean_squared_error, r2_score
mse = mean_squared_error(y_test, y_pred)

r_squared = r2_score(y_test, y_pred)
print('Mean_Squared_Error :' ,mse)
print('r_square_value :',r_squared)
