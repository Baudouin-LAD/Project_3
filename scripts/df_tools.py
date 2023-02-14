# -*- coding: utf-8 -*-
"""
Created on Mon Feb 13 15:20:59 2023

@author: marci
"""
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from typing import List, Optional
from scipy import stats

def load_data(pathfname,sep=","):
    data = pd.read_csv(pathfname)
    print(data.info())
    return data

def data_inspection(data: pd.DataFrame):
    print("Check for duplicated rows")
    duplidata = data[data.duplicated()]
    print(f"Duplicated rows found in original data: {duplidata.shape[0]}")
    print("Checking for null values")
    print(data.isnull().sum().sort_values(ascending=False))

def violent_crimes(data):
    data["violent_crimes"] = data["murders"] + data["rapes"] + data["robberies"] + data["assaults"]
    return data

def other_crimes(data):
    data["other_crimes"] = data["arsons"] + data["autoTheft"] + data["burglaries"] + data["larcenies"]
    return data

def check_outlier_IQR(data, column_name):
    ## Let's find out the outliers in `nonViolPerPop` response variable and plot those to find cities with highest crime rate in the US
    quartile_1, quartile_3 = np.percentile(data[column_name].dropna(), [25, 75])
    iqr = quartile_3 - quartile_1
    lower_bound = quartile_1 - (iqr * 1.5)
    upper_bound = quartile_3 + (iqr * 1.5)
    return data[(data[column_name]>upper_bound)|(data[column_name]<lower_bound)]
    
def check_outlier_upperIQR(data, column_name):
    ## Let's find out the outliers in `nonViolPerPop` response variable and plot those to find cities with highest crime rate in the US
    quartile_1, quartile_3 = np.percentile(data[column_name].dropna(), [25, 75])
    iqr = quartile_3 - quartile_1
    lower_bound = quartile_1 - (iqr * 1.5)
    upper_bound = quartile_3 + (iqr * 1.5)
    return data[data[column_name]>upper_bound]
    
def select_sample(category,data,indexcol:str, rankedcol = str, aggfunc="sum"):
    if category=='best':
        result=data.nsmallest(5,rankedcol)
    elif category=='worst':
        result=data.nlargest(5,rankedcol)
    else:
        print('invalid argument please select best or worst')
    return result

#Statistics

def selecting(data,crime_type,target,states):
    
    if crime_type=='other':
        target1=target.copy()
        target1.append('other_crimes_p_pop')
        if len(states) == 0:
            slice1=data.loc[:,target1]
        else: 
            target2=target1.copy() 
            target2.append('state')
            slice1=data.loc[data['state'].isin(states),target]
            
    elif crime_type =='violent':
        target3=target.copy()
        target3.append('violent_crimes_p_pop')
        if len(states) == 0:
            slice1=data.loc[:,target3]
        else: 
            target3.append('state')
            slice1=data.loc[data['state'].isin(states),target3]
    elif crime_type == 'all':
        target4=target.copy()
        target4.append('crime_p_pop')
        if len(states) == 0:
            slice1=data.loc[:,target4]
        else: 
            target4.append('state')
            slice1=data.loc[data['state'].isin(states),target4]
    
    return(slice1,crime_type,target,states)
    #creating an indice if targets>1
def indice(data,crime_type,target,states) :
    if len(target)>1:
        weights=[]
        n=0
        d=pd.DataFrame()
        for i in range (len(target)):
            j=int(input(f'what weight would you like to put on {target[i]}?'))
            n+=int(j)
            weights.append(j)
        for w in range(len(weights)):
            d[str(w)]=data[target[w]]*weights[w]
        data['indice']=d.sum(axis=1)/n
        return(data,crime_type,target,states)

def regress(data,crime_type,target,states): 
    X = sm.add_constant(data[target].iloc[:,1:].values)
    Y= data.iloc[:,-1].values
    model = sm.OLS(Y, X).fit()
    print_model = model.summary()
    #setting X and Y axis for plotting
    if len(target)>1:
        Xaxis=data.iloc[:,-3]
        Yaxis=data['indice']
    else:
        Xaxis=data.iloc[:,-2]
        Yaxis=data.iloc[:,0]
        
    return [data,crime_type,target,states,Xaxis,Yaxis,print_model]

# data = pd.read_csv('/Users/Baudouin/Ironhack/Project_4/data/crimedata.csv')
# states= pd.read_excel('/Users/Baudouin/Ironhack/Project_4/State_codes.xlsx')
# data = pd.merge(left=states, right=data, left_on='Alpha code', right_on='state')
# data["violent_crimes"] = data["murders"] +  data["rapes"] + data["robberies"] + data["assaults"]
# data["other_crimes"] = data["arsons"] +  data["autoTheft"] + data["burglaries"] + data["larcenies"]
# target_vars = ["population","murders",
# "rapes",
# "robberies",
# "assaults",
# "burglaries",
# "larcenies",
# "autoTheft",
# "arsons"]