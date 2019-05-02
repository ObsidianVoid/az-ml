from sklearn.linear_model import LinearRegression

from sklearn.datasets import load_boston

from sklearn.model_selection import train_test_split 

from sklearn.metrics import mean_squared_error


import os

import pickle

os.makedirs('./outputs', exist_ok=True)


X, y = load_boston(True)

regression = LinearRegression()

X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size = 0.25)

regression.fit(X_train, Y_train)

predicted = regression.predict(X_test)

mse = mean_squared_error(Y_test, predicted)

model_path = 'outputs/model.pkl'

with open(model_path, "wb") as file:
    from sklearn.externals import joblib
    joblib.dump(value=regression, filename=os.path.join('./outputs/', 'model.pkl'))