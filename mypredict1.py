# -*- coding: utf-8 -*-


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

def plot_data(dts, title, delta):
    print("inside")
    dts = dts.sort_values(by = ['date'], ascending =True)
    dts['x'] = np.arange(len(dts))+1
    dts['y'] = dts['total_cases']
    
    x = dts['x'][ :-delta]
    y = dts['y'][ :-delta]
    
    
    c2 = scipy.optimize.curve_fit(lambda t, a, b: a*np.exp(b*t),  x,  y,  p0=(20, 0.2)) 

    A, B = c2[0]  #Coefficients
    print(f'(y = Ae^(Bx)) A: {A}, B: {B}\n')
    x = range(1,dts.shape[0] + 1)
    y_fit = A * np.exp(B * x)
#    print(y_fit)
    f, ax = plt.subplots(1,1, figsize=(12,6))
    g = sn.scatterplot(x=dts['x'][:-delta], y=dts['y'][:-delta], label='Confirmed cases (used for model creation)', color='red')
    g = sn.scatterplot(x=dts['x'][-delta:], y=dts['y'][-delta:], label='Confirmed cases (not used for model, va;idation)', color='blue')
    g = sn.lineplot(x=x, y=y_fit, label='Predicted values', color='green')  #Predicted
    x_future=range((112),(117)) #As of 24 March 2020 we have 85 days of info. 
    y_future=A * np.exp(B * x_future)
    print("Expected cases for the next 5 days: \n", y_future)
    plt.xlabel('Days since first case')
    plt.ylabel(f'Total cases')
    plt.title(f'Confirmed cases & projected cases: {title}')
    plt.xticks(rotation=90)
    ax.grid(color='black', linestyle='dotted', linewidth=0.75)
    plt.show()
    
    
CVD_USA = dts_wo_china[dts_wo_china['location']=='India']


d_df = CVD_USA.copy()
plot_data(d_df, 'India', 5)
