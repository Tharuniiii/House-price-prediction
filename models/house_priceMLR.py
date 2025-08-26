
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
import os
import pickle

dataset = pd.read_csv(r"C:\Users\Tharuni\Desktop\NIT\Aug month\18th,19th-regression frontned backedn\house price prediction\data\House_data.csv")

# Independent variables: bedrooms, bathroom, sqft_living (columns 3,4,5)
x = dataset.iloc[:, 3:6]

# Dependent variable: price (column 2)
y = dataset.iloc[:, 2]

x = pd.get_dummies(x,dtype = int)

#split the data

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)


regressor = LinearRegression()
regressor.fit(x_train,y_train)

y_pred = regressor.predict(x_test)

m = regressor.coef_
print(m)

c = regressor.intercept_
print(c)

x = np.append(arr = np.full((21613,1),67326).astype(int),values = x,axis = 1)


x_opt = x[:,[0,1,3]]
regressor_OLS = sm.OLS(endog=y,exog =x_opt).fit()#endog-garbage in,exog = garbage out(fit in and fit out)
regressor_OLS.summary()

filename = 'multiple_linear_regression_model.pkl'
with open(filename, 'wb') as file:
    pickle.dump(regressor, file)
print("Model has been pickled and saved as multiple_linear_regression_model.pkl")

print("Full path:", os.path.abspath(filename))