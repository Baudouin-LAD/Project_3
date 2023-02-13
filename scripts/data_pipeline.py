# -*- coding: utf-8 -*-
"""
Created on Mon Feb 13 14:10:24 2023

@author: marci
"""

import os
os.getcwd()
os.chdir("c:\\Users\\marci\\dev\\Project_Week4\\scripts")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plot_tools
import df_tools

#%% READING DATABASE

data = pd.read_csv("../data/crimedata.csv")

print(data.head())

states= pd.read_excel("../State_codes.xlsx")
data = pd.merge(left=states, right=data, left_on='Alpha code', right_on='state')
    
#%% CHECKING ASPECTS FOR DATA CLEANING
df_tools.data_inspection(data)

#%%DEFINING VARIABLES OF INTEREST
target_vars =["murders",
                "rapes",
                "robberies",
                "assaults",
                "burglaries",
                "larcenies",
                "autoTheft",
                "arsons"]
columns_interest = target_vars.copy()
columns_interest.append("state")
columns_interest.append("population")
data["violent_crimes"] = data["murders"] +  data["rapes"] + data["robberies"] + data["assaults"]
data["other_crimes"] = data["arsons"] +  data["autoTheft"] + data["burglaries"] + data["larcenies"]
data["other_crimes"].fillna(0,inplace=True)
data["violent_crimes"].fillna(0,inplace=True)
data["crime"]=data["violent_crimes"]+data["other_crimes"]
#%% CREATING INDICATORS PER POPULATION
crime_p_state = data.pivot_table(index="State",values = columns_interest, aggfunc="sum")
crime_p_state = df_tools.violent_crimes(crime_p_state)
crime_p_state = df_tools.other_crimes(crime_p_state)
crime_p_state = pd.merge(left=states, right=crime_p_state, left_on='State', right_index=True)
crime_p_state.rename(columns={"Alpha code":"state"},inplace=True)

crime_p_state["violent_crimes_p_pop"] = crime_p_state["violent_crimes"]/crime_p_state["population"]
crime_p_state["other_crimes_p_pop"] = crime_p_state["other_crimes"]/crime_p_state["population"]
crime_p_state.sort_values(by="violent_crimes_p_pop",inplace=True,ascending=False)
#%% SELECT TOP AND WORST

least_violent = df_tools.select_sample(category ="best", data=crime_p_state,indexcol="State",
                       rankedcol="violent_crimes_p_pop" )

most_violent = df_tools.select_sample(category ="worst", data=crime_p_state,indexcol="State",
                        rankedcol="violent_crimes_p_pop" )
#%% DESCRIPTIVE ANALYSIS
fig, axes= plt.subplots(1,2,figsize=(10,5),sharey=True)
for nax,ax in enumerate(axes.flatten()):
    if nax == 0:
        plot_data = least_violent
        ax.title.set_text("Least violent")
    else:
        plot_data = most_violent
        ax.title.set_text("Most violent")
    plot_tools.barplot_pcat(ax, data=plot_data, xlevel='State', 
                            hue_vars=["violent_crimes_p_pop","other_crimes_p_pop"])
    plot_tools.addplotlabels(ax =ax, xlabel="State", ylabel="N° of crimes per pop.") 
    plt.sca(ax)
    plt.xticks(rotation = 30) # Rotates X-Axis Ticks by 45-degrees
fig.tight_layout()
#TODO
#Add fig save

#CREATING PIE CHART PER MOST AND LEAD VIOLENT STATES
plot_data = least_violent
axes = plot_tools.pie_chart(plot_data, ["murders","rapes","robberies","assaults"], ycol=None)
plot_data = most_violent
axes = plot_tools.pie_chart(plot_data, ["murders","rapes","robberies","assaults"], ycol=None)
#%% DATA SLICING

most_violent_data = df_tools.selecting(data, crime_type = "violent", states =most_violent.state, target=target_vars)

#%% PLOTTING PER VARIABLES OF INTERST 
def regression_plot(data, crime_type,target,states, plot_type):
    data["crime_p_pop"]=data["crime"]/data["population"]
    data["violent_crimes_p_pop"] = data["violent_crimes"]/data["population"]
    data["other_crimes_p_pop"] = data["other_crimes"]/data["population"]
    r = df_tools.selecting(data,crime_type,target,states)
    #r= df_tools.selecting(data,'all',['PctTeen2Par','PctYoungKids2Par'],['AL','AK','AZ','AR','CA','CO','CT','DE'])          
    s= df_tools.indice(r[0],r[1],r[2],r[3])
    t= df_tools.regress(s[0],s[1],s[2],s[3])  
    #return [data,crime_type,target,states,Xaxis,Yaxis,print_model]
    
    if plot_type == "regplot":
        fig, ax =plt.subplots(1,1)
        plot_tools.regplot_sns(ax=ax, data=data, xcol = t[4], ycol = t[5])
        plt.show()
    return t[6]
regression_plot(data, 'all',['PctTeen2Par','PctYoungKids2Par'],most_violent.state,"regplot")
#%% ANALYSE RELATION OF VARIABLES WE'RE INTERESTED IN
# ydata = "PolicPerPop"
# sns.boxplot()
 
 
# fig,ax=plt.subplots(1,1)
# plot_tools.regplot_sns(data=df,ax=ax,ycol="murders",xcol="PolicPerPop")
