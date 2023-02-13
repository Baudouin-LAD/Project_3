#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 13 09:31:29 2023

@author: Baudouin
"""

import pandas as pd

data = pd.read_csv('/Users/Baudouin/Ironhack/Project_4/data/crimedata.csv')
target_vars = ["population","murders",
"rapes",
"robberies",
"assaults",
"burglaries",
"larcenies",
"autoTheft",
"arsons"]


crime_p_state=data.pivot_table(index='state',values=target_vars,aggfunc ='sum')
crime_p_state["violent_crimes"] = crime_p_state["murders"] +  crime_p_state["rapes"] + crime_p_state["robberies"] + crime_p_state["assaults"]
crime_p_state["other_crimes"] = crime_p_state["arsons"] +  crime_p_state["autoTheft"] + crime_p_state["burglaries"] + crime_p_state["larcenies"]
crime_p_state["violent_crimes_p_pop"] = crime_p_state["violent_crimes"]/crime_p_state["population"]
crime_p_state["other_crimes_p_pop"] = crime_p_state["other_crimes"]/crime_p_state["population"]
most_violent=crime_p_state.nlargest(5,'violent_crimes_p_pop')