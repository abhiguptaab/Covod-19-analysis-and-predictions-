# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import datetime as dt
import matplotlib

dataset = pd.read_csv("owid-covid-data.csv")
dataset['date'] = [dt.datetime.strptime(x, '%Y-%m-%d') for x in dataset['date']]

country = ['India', 'Italy', 'Spain', 'United States']
dataset_country = dataset[dataset.location.isin(country)]

dataset_country.set_index('date', inplace = True)
dataset_country['mortality_rate'] = dataset_country['total_deaths']/dataset_country['total_cases']*100

fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(14,14))

gf = dataset_country.groupby('location')['new_cases'].plot(ax = axes[0,0], legend = True)
dataset_country.groupby('location')['new_deaths'].plot(ax = axes[0,1], legend = True)
dataset_country.groupby('location')['total_cases'].plot(ax = axes[1,0], legend = True)
dataset_country.groupby('location')['total_deaths'].plot(ax = axes[1,1], legend = True)

axes[0,0].set_title("new cases")
axes[0,1].set_title("new_deaths")
axes[1,0].set_title("total_cases")
axes[1,1].set_title("total_deaths")

print(dataset_country.isnull().sum())

dts1 = pd.DataFrame(dataset.groupby(['location', 'date'])['total_cases','total_deaths'].sum()).reset_index()
dts = dts1.sort_values(by = ['location', 'date'], ascending = False)

import seaborn as sn

filtered_dts = dts.drop_duplicates(subset = ['location'], keep = 'first')
def barplot(feature, values, title, ds, size):
    f, axes = plt.subplots(1,1,  figsize=(4*size,4))
    ds = filtered_dts.sort_values([values], ascending = False).reset_index(drop=True)
    g = sn.barplot(ds[feature][0:20], ds[values][0:20], palette = 'Set3')
    g.set_title("Number of {} - heghest 10 values".format(title))
    plt.show()

barplot('location', 'total_cases', 'total cases in te world', filtered_dts , 4)
barplot('location', 'total_deaths', 'total deaths in te world', filtered_dts , 4)
    

dts_agg = dts.groupby(['date']).sum().reset_index()

def plot_world_aggregate(df, title, size):
    f , ax = plt.subplots(1,1, figsize=(4*size, 2*size))
    g = sn.lineplot(x='date', y='total_cases', data = df, color = 'blue', label = 'total cases')
    g = sn.lineplot(x='date', y='total_deaths', data= df, color = 'red', label = 'total deaths')
    
    plt.xlabel('Date')
    plt.ylabel(f'Total {title} cases')
    plt.xticks(rotation=90)
    plt.title(f'Total {title} cases')
    ax.grid(color='black', linestyle='dotted', linewidth=0.75)
    plt.show()   
plot_world_aggregate(dts_agg, 'world map', 4)


dts['mortality_rate'] = dts['total_deaths']/dts['total_cases']*100
dts_mortality_sorted = dts.sort_values(by=['location', 'date'], ascending = False).reset_index()
#dts_mortality_sorted1 = pd.DataFrame(dts_mortality_sorted.groupby(['date','location'])['mortality_rate'])
dts_mortality_sorted1=dts_mortality_sorted.drop_duplicates(subset=['location'], keep = 'first')

#dts_mortality_sorted1['location','mortality_rate'] = dts_mortality_sorted['location','mortality_rate']
#dts_mortality_sorted1 = pd.DataFrame(dts_mortality_sorted.groupby(['location'])['mortality_rate']).reset_index()
def mortality_rate(feature, values, title, ds, size):
    f, axes = plt.subplots(1,1,  figsize=(10*size,4))
    ds = dts_mortality_sorted1.sort_values([values], ascending = False).reset_index(drop=True)
    g = sn.barplot(ds[feature][0:20], ds[values][0:20], palette = 'Set3')
    g.set_title("Number of {} - heghest 10 values".format(title))
    plt.show()
    
mortality_rate('location', 'mortality_rate', 'mortality rate of countries ', dts_mortality_sorted1  , 4)