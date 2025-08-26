import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import os 
import pickle 

#load the dataset
dataset = pd.read_csv(r"C:\Users\Tharuni\Desktop\NIT\Aug month\18th,19th-regression frontned backedn\house price prediction\House_data.csv")
space=dataset['sqft_living']
price=dataset['price']
#split the variables to dependent and independent 
x = np.array(space).reshape(-1, 1)
y = np.array(price)

#data splitting for testing and training
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)

#train the model
Regressor = LinearRegression()
Regressor.fit(x_train,y_train)

#predict the test data
y_pred = Regressor.predict(x_test)

#Visualization of predicted train data
plt.scatter(x_train,y_train, color = 'red')
plt.plot(x_train, Regressor.predict(x_train),color = 'blue')
plt.title('space vs price (training data)')
plt.xlabel('house space sqrt')
plt.ylabel('house price')
plt.show()

#Visualization of predicted test data
plt.scatter(x_test,y_test, color = 'red')
plt.plot(x_train, Regressor.predict(x_train),color = 'blue')
plt.title('space vs price (testing data)')
plt.xlabel('house space sqrt')
plt.ylabel('house price')
plt.show()

#predict the price of house with 3000 and 5000 space
y_350 = Regressor.predict([[350]])
y_550 = Regressor.predict([[550]])
print(f"predicted price for 300 space: ${float (y_350[0]):,.2f}")
print(f"predicted price for 500 space: ${float (y_550[0]):,.2f}")

#check model performance
bias = Regressor.score(x_train, y_train)
variance = Regressor.score(x_test, y_test)
train_mse = mean_squared_error(y_train, Regressor.predict(x_train))
test_mse = mean_squared_error(y_test, y_pred)

print(f"Training Score (R^2): {bias:.2f}")
print(f"Testing Score (R^2): {variance:.2f}")
print(f"Training MSE: {train_mse:.2f}")
print(f"Test MSE: {test_mse:.2f}")

# Save the trained model to disk
filename = 'simple_linear_regression_model.pkl'
with open(filename, 'wb') as file:
    pickle.dump(Regressor, file)
print("Model has been pickled and saved as simple_linear_regression_model.pkl")

print("Full path:", os.path.abspath(filename))