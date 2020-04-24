# -*- coding: utf-8 -*-

# -*- coding: utf-8 -*-

import sklearn
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import datetime as dt
import pandas as pd
import seaborn as sn
import scipy
dataset = pd.read_csv("owid-covid-data.csv")
dataset['date'] = [dt.datetime.strptime(x, '%Y-%m-%d') for x in dataset['date']]

dts_wo_china = dataset.loc[~(dataset['location'].isin(['china', 'world']))]

dts_wo_china = pd.DataFrame(dts_wo_china.groupby(['location', 'date'])['total_cases', 'total_deaths'].sum()).reset_index()

dts_wo_china = dts_wo_china.sort_values(by = ['location', 'date'], ascending =False)

dts_wo_china = dts_wo_china[dts_wo_china['location']=='India']
print("inside")
dts_wo_china = dts_wo_china.sort_values(by = ['date'], ascending =True)

dts_wo_china['x'] = np.arange(len(dts_wo_china))+1
dts_wo_china['y'] = dts_wo_china['total_cases']
dts_wo_china['z'] = dts_wo_china['total_deaths']

x = dts_wo_china.iloc[:, 4:5].values
y = dts_wo_china.iloc[:, 5:6].values
z = dts_wo_china.iloc[:, 6:7].values

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
poly_regression = PolynomialFeatures(degree=7)
X_poly = poly_regression.fit_transform(x)


lin_regression_2 = LinearRegression()
lin_regression_2.fit(X_poly, y)
poly_pred_Y = lin_regression_2.predict(X_poly)


poly_regression = PolynomialFeatures(degree=7)
X_poly = poly_regression.fit_transform(x)
lin_regression_3 = LinearRegression()
lin_regression_3.fit(X_poly, z)
poly_pred_z = lin_regression_3.predict(X_poly)

#from sklearn.metrics import classification_report, confusion_matrix  #print(classification_report(y, poly_pred_Y))

#print(lin_regression_2.predict(poly_regression.fit_transform([[120]])))

print("\n\n \t\tDay 1 is started from 2019-12-31 \n")
print("\n\n \t\tInput number of day at which you want to predict cases ofr deaths \n")
day = input()



plt.scatter(x,y, color = "red")
plt.plot(x, poly_pred_Y, color= "blue")
plt.plot(x, poly_pred_Y, color= "blue")
plt.title("Total cases prediction in India")
plt.xlabel("day")
plt.ylabel("cases")
img =plt.show()
plt.savefig("Total cases prediction.png")
print("\n\n \t\tTotal number of predicted cases \n")
print(lin_regression_2.predict(poly_regression.fit_transform([[day]])))

plt.scatter(x,z, color = "red")
plt.plot(x, poly_pred_z, color= "blue")
#plt.plot(x, poly_pred_Y, color= "blue")
plt.title("Death prediction in india")
plt.xlabel("day")
plt.ylabel("cases")
plt.show()
plt.savefig("Death_prediction.png")
print("\n\n \t\tTotal number of predicted deaths \n")
print(lin_regression_3.predict(poly_regression.fit_transform([[day]])))