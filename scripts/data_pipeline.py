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

plt.rcParams.update({'font.size': 16})

#%% READING DATABASE

data = pd.read_csv("../data/crimedata.csv")

print(data.head())
data.info()

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
data["crime_p_pop"]=data["crime"]/data["population"]
data["violent_crimes_p_pop"] = data["violent_crimes"]/data["population"]
data["other_crimes_p_pop"] = data["other_crimes"]/data["population"]
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
    r = df_tools.selecting(data,crime_type,target,states)
    #r= df_tools.selecting(data,'all',['PctTeen2Par','PctYoungKids2Par'],['AL','AK','AZ','AR','CA','CO','CT','DE'])          
    s= df_tools.indice(r[0],r[1],r[2],r[3])
    t= df_tools.regress(s[0],s[1],s[2],s[3])  
    #return [data,crime_type,target,states,Xaxis,Yaxis,print_model]
    
    if plot_type == "regplot":
        fig, ax =plt.subplots(1,1)
        plot_tools.regplot_sns(ax=ax, data=data, xcol = t[5], ycol = t[4])
        plt.show()
    return t[6], ax
regoutput, ax = regression_plot(data, 'all',['PctTeen2Par','PctYoungKids2Par'],most_violent.state,"regplot")
plot_tools.addplotlabels(ax, "Family situation Index", "N° of crimes per person")
#%% ANALYSE RELATION OF VARIABLES WE'RE INTERESTED IN
import seaborn as sns
yvar = "PolicPerPop"
xvar = "State"
plot_data= df_tools.selecting(data, crime_type = "violent", states =most_violent.state, target=[xvar,yvar])[0]


radar_plot = plot_tools.radar_chart(data, target_vars, most_violent.state, "Distribution of crimes for most violent states")
radar_plot.savefig("../img/radar_plot_mostviolentstates.png")
# fig,ax=plt.subplots(1,1)
# plot_tools.regplot_sns(data=df,ax=ax,ycol="murders",xcol="PolicPerPop")
#%%
print("Interest family composition related variables")
vars_interest = [
"MalePctDivorce",
"MalePctNevMarr",
"FemalePctDiv",
"TotalPctDiv",
"PersPerFam",
"PctFam2Par",
"PctKids2Par",
"PctKids2Par_Non",
"PctYoungKids2Par",
"PctTeen2Par",
"PctWorkMomYoungKids"]
dep_vars = ["ViolentCrimesPerPop", "nonViolPerPop"]
# dep_vars = ['murdPerPop',
# 'rapesPerPop','robbbPerPop','assaultPerPop','ViolentCrimesPerPop',
# 'burglPerPop','larcPerPop','autoTheftPerPop','arsonsPerPop','nonViolPerPop']
vars_interest.extend(dep_vars)

#vars_interest.append("violent_crimes_p_pop")
data["PctKids2Par_Non"] = 1 - data["PctKids2Par"] 

corr_data, ax = plot_tools.corr_heatmap(data, vars_interest)


most_corr_data = corr_data.loc[:,dep_vars]
most_corr_data.sort_values(by="ViolentCrimesPerPop",ascending=False,inplace=True)
#%% Focused on most correlated vars
vars_interest = [
"PctKids2Par_Non",
"PctFam2Par",
"PctYoungKids2Par",
"PctTeen2Par",
"FemalePctDiv",
"TotalPctDiv"]
dep_vars = ['murdPerPop',
'rapesPerPop','robbbPerPop','assaultPerPop','ViolentCrimesPerPop',
'burglPerPop','larcPerPop','autoTheftPerPop','arsonsPerPop','nonViolPerPop']
vars_interest.extend(dep_vars)
corr_data, ax = plot_tools.corr_heatmap(data, vars_interest)

#%%
print("Household related")
vars_interest = ['HousVacant','PctHousOccup','PctHousOwnOcc','PctVacantBoarded',
        'PctVacMore6Mos','PctUnemployed','PctEmploy']

vars_interest
dep_vars = ["ViolentCrimesPerPop", "nonViolPerPop"]
vars_interest.extend(dep_vars)

corr_data, ax = plot_tools.corr_heatmap(data, vars_interest)


#%%
print("Household related in detail per crime")
vars_interest = ['HousVacant','PctHousOccup','PctHousOwnOcc','PctVacantBoarded',
        'PctVacMore6Mos','PctUnemployed','PctEmploy']

dep_vars = ['murdPerPop',
'rapesPerPop','robbbPerPop','assaultPerPop','ViolentCrimesPerPop',
'burglPerPop','larcPerPop','autoTheftPerPop','arsonsPerPop','nonViolPerPop']
vars_interest.extend(dep_vars)
corr_data, ax = plot_tools.corr_heatmap(data, vars_interest)
#%%
cols = ['HousVacant','PctHousOccup','PctHousOwnOcc','PctVacantBoarded',
        'PctVacMore6Mos','PctUnemployed','PctEmploy','murdPerPop','rapesPerPop',
        'robbbPerPop','assaultPerPop','ViolentCrimesPerPop','burglPerPop','larcPerPop',
        'autoTheftPerPop','arsonsPerPop','nonViolPerPop']
crimedata_study = data.filter(cols, axis=1)
corr_crimedata_study = crimedata_study.corr()
iv_corr = corr_crimedata_study.iloc[:-10,:-10]
multicol_limit = 0.3
iv_corr = (iv_corr[abs(iv_corr) > multicol_limit][iv_corr != 1.0]).unstack().to_dict()
iv_multicoll_corr = pd.DataFrame(list(set([(tuple(sorted(key)), iv_corr[key]) for key in iv_corr])), 
        columns=['Independent Variables', 'Correlation Coefficient'])
print (iv_multicoll_corr[iv_multicoll_corr.notnull().all(axis=1)])

#%%CREATING USA MAP WITH A Z LEVEL according to variable
import plotly.io as io
io.renderers.default='browser'
#io.renderers.default='svg'



import plotly
import plotly.graph_objs as go
import math #needed for definition of pi

# xpoints = np.arange(0, math.pi*2, 0.05)
# ypoints = np.sin(xpoints)
# trace0 = go.Scatter(
#    x = xpoints, y = ypoints
# )
# test = [trace0]
# #when jupyter notebook iplot
# plotly.offline.plot({ "data": test,"layout": go.Layout(title="Sine wave")
#                            }, image_width=580, image_height=580,  
#              filename='temp-plot.html', auto_open=False)
# plotly.io.write_image(renderer="svg")


##Aggregate view of Non-Violent Crimes by State
plot_tools.plotly_map_usa(data, "murders", vmin=0,vmax=50,
                          title="N° or murders", colorbartitle="N°",
                          filename="../img/murders_absolute.html", image_width=600,
                          image_height=400)

plot_tools.plotly_map_usa(data, "ViolentCrimesPerPop", vmin=0,vmax=2500,
                          title="Violent Crimes Per 100K", colorbartitle="N° crimes",
                          filename="../img/ViolentCrimesPerPop.html", image_width=600,
                          image_height=400)
plot_tools.plotly_map_usa(data, "nonViolPerPop", vmin=0,vmax=15000,
                          title="Non violent crimes per 100K", colorbartitle="N° crimes",
                          filename="../img/nonViolentCrimesPerPop.html", image_width=600,
                          image_height=400)
#violent_crimes_p_pop
plot_tools.plotly_map_usa(data, "violent_crimes_p_pop", vmin=0,vmax=1,
                          title="Violent Crimes Per person", colorbartitle="N° crimes p per",
                          filename="../img/ViolentCrimesPerPop_indv.html", image_width=600,
                          image_height=400)

#%% CREATE GEOREFERENCED DATA
## read citie.json file to get latitude and longitude details of the cities
import json
filename = "../data/cities.json"
with open(filename) as city_file:
   dict_city = json.load(city_file)
cities_lat_lon = pd.json_normalize(dict_city)

#%% OUTLIERS PER CITY
import re
## Let's find out the outliers in `nonViolPerPop` response variable and plot those to find cities with highest crime rate in the US
outliers_nviol = df_tools.check_outlier_upperIQR(data, 'nonViolPerPop')
outliers_viol = df_tools.check_outlier_upperIQR(data, 'ViolentCrimesPerPop')


## Remove community Name(s) ending with "city". This helps dataframe(s) merging easier to get lat and lon
outliers_viol['communityName'] = outliers_viol['communityName'].map(lambda result : re.sub(r'city','',result))
outliers_nviol['communityName'] = outliers_nviol['communityName'].map(lambda result : re.sub(r'city','',result))

violent_crime_cities = pd.merge(outliers_viol,cities_lat_lon,left_on=["communityName","state"],right_on=["city","state"])
violent_crime_cities = violent_crime_cities.drop(["city"],axis=1)
#print (violent_crime_cities)

nonviolent_crime_cities = pd.merge(outliers_nviol,cities_lat_lon,left_on=["communityName","state"],right_on=["city","state"])
nonviolent_crime_cities = nonviolent_crime_cities.drop(["city"],axis=1)

## Cities with highest  non-violent crime rate
levels = [(0,20),(21,30),(31,40),(41,50),(51,80)]
colors = ['rgb(255,133,27)','rgb(31,120,180)','rgb(178,223,138)','rgb(251,154,153)','rgb(227,26,28)']
plot_data = []
for i in range(len(levels)):
    lim = levels[i]
    nonviolent_crime_cities_sub = nonviolent_crime_cities[lim[0]:lim[1]]
    city_outline = dict(
        type = "scattergeo",
        locationmode = 'USA-states',
        lon = nonviolent_crime_cities_sub['longitude'],
        lat = nonviolent_crime_cities_sub['latitude'],
        text = nonviolent_crime_cities_sub['communityName'] +' '+ nonviolent_crime_cities_sub['nonViolPerPop'].astype(str),
        mode = "markers",
        marker = dict(
        size = nonviolent_crime_cities_sub['nonViolPerPop']/800,
        color = colors[i],
        ),  
    name = '{0} - {1}'.format(lim[0],lim[1])
    )
    layout1 = dict(
        title = 'Cities with highest non-Violent Crime rate',
        geo = dict(
            scope='usa',
            projection=dict( type='albers usa' ),
            showlakes = True,
            lakecolor = 'rgb(85,173,240)'), 
    )
    plot_data.append(city_outline)
    
fig1= dict( data=plot_data, layout=layout1)
plotly.offline.plot( fig1, validate=False,
                    filename="../img/nonviolentoutliers.html", auto_open=True)## Cities with highest violent crime rate
levels = [(0,30),(31,60),(61,90),(91,120),(121,170)]
colors = ['rgb(175,175,50)','rgb(131,120,180)','rgb(78,230,138)','rgb(251,24,153)','rgb(227,126,28)']
cities = []
for i in range(len(levels)):
    lim = levels[i]
    violent_crime_cities_sub = violent_crime_cities[lim[0]:lim[1]]
    city_outline = dict(
        type = "scattergeo",
        locationmode = 'USA-states',
        lon = violent_crime_cities_sub['longitude'],
        lat = violent_crime_cities_sub['latitude'],
        text = violent_crime_cities_sub['communityName'] +' '+ violent_crime_cities_sub['ViolentCrimesPerPop'].astype(str),
        mode = "markers",
        marker = dict(
        size = violent_crime_cities_sub['ViolentCrimesPerPop']/200,
        color = colors[i]
        ),  
    name = '{0} - {1}'.format(lim[0],lim[1])
    )
    layout2 = dict(
        title = 'Cities with highest Violent Crime rate',
        geo = dict(
            scope='usa',
            projection=dict( type='albers usa' ),
            showlakes = True,
            lakecolor = 'rgb(85,173,240)'),
    ) 
    cities.append(city_outline)
    
fig2= dict( data=cities, layout=layout2)
plotly.offline.plot( fig2, validate=False,
                    filename="../img/violentoutliers.html", auto_open=True)