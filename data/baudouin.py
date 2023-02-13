#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 13 09:31:29 2023

@author: Baudouin
"""

import pandas as pd

data = pd.read_csv('/Users/Baudouin/Ironhack/Project_4/data/crimedata.csv')
states= pd.read_excel('/Users/Baudouin/Ironhack/Project_4/State_codes.xlsx')
data = pd.merge(left=states, right=data, left_on='Alpha code', right_on='state')

target_vars = ["population","murders",
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


r=select_sample('best',data)
