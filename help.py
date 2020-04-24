# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 12:58:10 2020

@author: Lenovo
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv("Position_Salaries.csv")

X = dataset.iloc[:, 1:2].values
Y = dataset.iloc[:, 2:3].values

#fitting linear regression
from sklearn.linear_model import LinearRegression
lin_regression = LinearRegression()
lin_regression.fit(X,Y)
lin_pred_Y = lin_regression.predict(X)

#fitting polynomial regression
from sklearn.preprocessing import PolynomialFeatures
poly_regression = PolynomialFeatures(degree=5)
X_poly = poly_regression.fit_transform(X)

lin_regression_2 = LinearRegression()
lin_regression_2.fit(X_poly, Y)
poly_pred_Y = lin_regression_2.predict(X_poly)

#visualizing linear regression model
plt.scatter(X,Y, color = 'red')
plt.plot(X,lin_pred_Y, color = "blue" )
plt.title("Truth or Buff(Linear regression)")
plt.xlabel("position level")
plt.ylabel("salary")
plt.show()

#visualizing polynomial model
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X,Y, color = "red")
plt.plot(X_grid, lin_regression_2.predict(poly_regression.fit_transform(X_grid)), color= "blue")
plt.title("Truth or Buff(Polynomial Linear regression)")
plt.xlabel("position level")
plt.ylabel("salary")
plt.show()

print(lin_regression.predict([[6.5]]))

# Predicting a new result with Polynomial Regression
print(lin_regression_2.predict(poly_regression.fit_transform([[6.5]])))# -*- coding: utf-8 -*-

