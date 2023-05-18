import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_squared_error

# Step 1: Load the dataset
dataset_path = "numbers.csv"
data = pd.read_csv(dataset_path)

# Step 2: Split the dataset into input features (X) and target variable (Y)
X = data[['AT', 'V', 'AP', 'RH']].values
Y = data['PE'].values

# Step 3: Split the data into training and testing sets
Xtrain, Xtest, ytrain, ytest = train_test_split(X, Y, test_size=0.2, random_state=42)

# Step 4: Create and fit your regression model
regr = LinearRegression()
regr.fit(Xtrain, ytrain)

# Step 5: Make predictions on the test data using your regression model
pred_test = regr.predict(Xtest)

# Step 6: Calculate the mean squared error (MSE) on the test data
mse_test = mean_squared_error(ytest, pred_test)
print("Test MSE (Regression):", mse_test)

# Step 7: Create and fit the dummy regressor model
dummy_regr = DummyRegressor(strategy='mean')
dummy_regr.fit(Xtrain, ytrain)

# Step 8: Make predictions on the test data using the dummy regressor
dummy_pred_test = dummy_regr.predict(Xtest)

# Step 9: Calculate the mean squared error (MSE) on the test data using the dummy regressor
dummy_mse_test = mean_squared_error(ytest, dummy_pred_test)
print("Test MSE (Dummy):", dummy_mse_test)

# Step 10: Compare the performance of your regression model and the dummy regressor
improvement = (dummy_mse_test - mse_test) / dummy_mse_test
print("Improvement over Dummy: {:.2%}".format(improvement))
