#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 13 09:31:29 2023

@author: Baudouin
"""

import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from typing import List, Optional
from scipy import stats

data = pd.read_csv('/Users/Baudouin/Ironhack/Project_4/data/crimedata.csv')
states= pd.read_excel('/Users/Baudouin/Ironhack/Project_4/State_codes.xlsx')
data = pd.merge(left=states, right=data, left_on='Alpha code', right_on='state')
data["violent_crimes"] = data["murders"] +  data["rapes"] + data["robberies"] + data["assaults"]
data["other_crimes"] = data["arsons"] +  data["autoTheft"] + data["burglaries"] + data["larcenies"]
data["crime"]=data["violent_crimes"]+data["other_crimes"]
target_vars = ["population",
"murders",
"rapes",
"robberies",
"assaults",
"burglaries",
"larcenies",
"autoTheft",
"arsons"]

def select_sample(category,data):
    crime_p_state=data.pivot_table(index='State',values=target_vars,aggfunc ='sum')
    crime_p_state["violent_crimes"] = crime_p_state["murders"] +  crime_p_state["rapes"] + crime_p_state["robberies"] + crime_p_state["assaults"]
    crime_p_state["other_crimes"] = crime_p_state["arsons"] +  crime_p_state["autoTheft"] + crime_p_state["burglaries"] + crime_p_state["larcenies"]
    crime_p_state["violent_crimes_p_pop"] = crime_p_state["violent_crimes"]/crime_p_state["population"]
    crime_p_state["other_crimes_p_pop"] = crime_p_state["other_crimes"]/crime_p_state["population"]
    if category=='best':
        result=crime_p_state.nsmallest(5,'violent_crimes_p_pop')
    elif category=='worst':
        result=crime_p_state.nlargest(5,'violent_crimes_p_pop')
    else:
        print('invalid argument please select best or worst')
    return result

#Statistics
"""def indice(data,target: List[str]):
    weights=[]
    n=O
    d=pd.DataFrame()
    for i in range (len(target)):
        j=input('what weight would you like to put on {target[i]}')
        n+=j
        weights.append(j)
    for w in range(len(weights)):
        d[str(w)]=data[target[w]]*weights[w]
    data['indice']=d.sum(axis=1)/n
    return[data,target]"""

        
def slicing(data,crime_type:str,states: List[str],target: List[str],Xaxis:Optional[str]=None,Yaxis:Optional[str]=None):
    if crime_type == 'violent':
        target.append('violent_crimes')
        if len(states) == 0:
            result=data.loc[:,target]
        else: 
            target.append('state')
            result=data.loc[data['state'].isin(states),target]
    if crime_type == 'other':
        target.append('other_crimes')
        if len(states) == 0:
            result=data.loc[:,target]
        else: 
            target.append('state')
            result=data.loc[data['state'].isin(states),target]
            result = [data,crime_type,states,target,data.iloc[:,-2],data['indice']]
    return result.reset_index().drop('index',axis=1)
            

r=slicing(data,'other',['IL','FL','DC'],['PctKids2Par','PctTeen2Par'])   

"""def regress(data,crime_type:Optional[str]=None,states: Optional[List[str]]=None,target: Optional[List[str]]=None,Xaxis:Optional[str]=None,Yaxis:Optional[str]=None):
    constants=[data.iloc[:,:-2]]
    if len(target)>1:
        data=indice(data,target)[0]
        X = sm.add_constant(data[constants])
        Y= data[]
        model = sm.OLS(Y, X).fit()
        predictions = model.predict(X) 
r=slicin
r=indice(data,['PctKids2Par','PctTeen2Par'])
print_model = model.summary()
print(print_model)
        
        result = [data,crime_type,states,target,data.iloc[:,-2],data['indice']]
    return result"""


def selecting(data,crime_type,target,states):
    
    if crime_type=='other':
        target1=target.copy()
        target1.append('other_crimes')
        if len(states) == 0:
            slice1=data.loc[:,target1]
        else: 
            target2=target1.copy() 
            target2.append('state')
            slice1=data.loc[data['state'].isin(states),target]
            
    elif crime_type =='violent':
        target3=target.copy()
        target3.append('violent_crimes')
        if len(states) == 0:
            slice1=data.loc[:,target3]
        else: 
            target3.append('state')
            slice1=data.loc[data['state'].isin(states),target3]
    elif crime_type == 'all':
        target4=target.copy()
        target4.append('crime')
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
        Xaxis=data.iloc[:,-1]
        Yaxis=data['indice']
    else:
        Xaxis=data.iloc[:,-1]
        Yaxis=data.iloc[:,0]
        
    return [data,crime_type,target,states,Xaxis,Yaxis,print_model]

    
r= selecting(data,'all',['PctTeen2Par','PctYoungKids2Par'],['AL','AK','AZ','AR','CA','CO','CT','DE'])          
s=indice(r[0],r[1],r[2],r[3])
t=regress(s[0],s[1],s[2],s[3])        
   
X = sm.add_constant(s[0][s[2]].values)
Y= s[0].iloc[:,-1].values
            
            
            
            
            
            
            
            
            
            
            