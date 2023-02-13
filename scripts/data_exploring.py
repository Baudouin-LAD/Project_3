# %%
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plot_tools

# %%
os.getcwd()
os.chdir("c:\\Users\\marci\\dev\\Project_Week4")

# %%
df = pd.read_csv("data/crimedata.csv")

# %%
df.head()

# %%
df.dtypes.to_csv("data/colums_info.csv")
df.dtypes

# %%
df.info()

# %%
for col in df.columns:
    print(col)

# %% [markdown]
# ### Cleaning data

# %% [markdown]
# #### Checking for duplicates

# %%
df[df.duplicated()]

# %%
df["PolicPerPop"].isnull().sum()

# %% [markdown]
# Grouped by State

# %%
target_vars = ["population","murders",
"rapes",
"robberies",
"assaults",
"burglaries",
"larcenies",
"autoTheft",
"arsons"]

# %%
columns_interest = target_vars.append("state")

# %%
data_state = df.groupby("state").sum()

# %%
os.getcwd()

# %%
states= pd.read_excel("State_codes.xlsx")
data = pd.merge(left=states, right=df, left_on='Alpha code', right_on='state')

# %%
data

# %%
crime_p_state = data.pivot_table(index="State", values = target_vars, aggfunc="sum")

# %%
crime_p_state["violent_crimes"] = crime_p_state["murders"] +  crime_p_state["rapes"] + crime_p_state["robberies"] + crime_p_state["assaults"]
crime_p_state["other_crimes"] = crime_p_state["arsons"] +  crime_p_state["autoTheft"] + crime_p_state["burglaries"] + crime_p_state["larcenies"]
crime_p_state["violent_crimes_p_pop"] = crime_p_state["violent_crimes"]/crime_p_state["population"]
crime_p_state["other_crimes_p_pop"] = crime_p_state["other_crimes"]/crime_p_state["population"]

# %%
crime_p_state.sort_values(by="violent_crimes_p_pop",inplace=True,ascending=False)

# %%
crime_p_state

# %%
worst_states = crime_p_state.nlargest(columns="violent_crimes_p_pop",n=5)
worst_states

# %%
best_states = crime_p_state.nsmallest(columns="violent_crimes_p_pop",n=5)
best_states

# %%
worst_states_df = worst_states.loc[:, ["violent_crimes_p_pop","other_crimes_p_pop"]]
worst_states_df= worst_states_df.merge(data, left_index=True, right_on="State")
worst_states_df

# %%
best_states_df =best_states.loc[:, ["violent_crimes_p_pop","other_crimes_p_pop"]]
best_states_df=best_states_df.merge(data, left_index=True, right_on="State")
best_states_df

# %%
target_vars = ["murders",
"rapes",
"robberies",
"assaults",
"burglaries",
"larcenies",
"autoTheft",
"arsons"]

# %%
os.chdir("c:\\Users\\marci\\dev\\Project_Week4\\scripts")
os.getcwd()

# %% [markdown]
# ### PLOT FUNCTIONS

# %%
from typing import List, Optional
import plot_tools

# %%
def heatmeap_sns(ax : plt.Axes, data: pd.DataFrame, mask=True, vmax=1, vmin=-1):
    # Generate a mask for the upper triangle
    mask = np.triu(np.ones_like(data, dtype=bool))
    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(data, ax=ax, mask=mask, cmap=cmap, vmax=vmax, vmin=vmin, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})
    return ax

# %%
def regplot_sns(ax : plt.Axes, data: pd.DataFrame, xcol:str, ycol:str):
    sns.regplot(data=data, x=xcol, y=ycol,  ax=ax)
    return ax
def addplotlabels(ax : plt.Axes, xlabel:str, ylabel:str) -> plt.Axes:
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    return ax

def boxplot(ax, data : pd.DataFrame,  xtitle : Optional[str] = None ):
#Boxplot of Violent crime variables
    sns.boxplot(data=data, ax=ax)
    return ax

def scatterplot(ax, df, xcol, ycol):
    ax.scatter(x=df[xcol],y=df[ycol])
    return ax

def histogram(ax,values, nbins=10):
    ax.hist(values, nbins=nbins)
    return ax

def pie_chart(ax,df, colname):
    df.plot.pie(y=colname,ax=ax)
    return ax

def scatterplot_sns(ax : plt.Axes, data: pd.DataFrame, xcol:str, ycol:str, hue_col : Optional[str] = None):
    sns.scatterplot(data=data, x=xcol, y=ycol, hue=hue_col, ax=ax)
    return ax
def barplot_pcat(ax, data: pd.DataFrame, xlevel:str, hue_vars: List[str] ):
    dfm = data.reset_index(drop=False)
    list_colnames = []
    list_colnames.append(xlevel)
    list_colnames.extend(hue_vars)
    dfm = dfm.loc[:, list_colnames]
    dfm = dfm.melt(id_vars=xlevel)
    sns.barplot(data=dfm, x=xlevel, y='value', hue="variable",ax=ax)
    return ax

# %%
fig, axes = plt.subplots(1,2, figsize=(12,5),sharey=True)
barplot_pcat(axes.flatten()[0], data=best_states, xlevel='State', hue_vars=["violent_crimes_p_pop","other_crimes_p_pop"])
barplot_pcat(axes.flatten()[1], data=worst_states, xlevel='State', hue_vars=["violent_crimes_p_pop","other_crimes_p_pop"])


# %%
plot_data = best_states.loc[:,["murders","rapes","robberies","assaults"]].reset_index(drop=False)
plot_data = plot_data.iloc[[0],:]
plot_data.plot

# %%
corr_best = best_states_df.corr()
fig, ax=plt.subplots(1,1)
heatmeap_sns(ax=ax,data=corr_best)

# %% [markdown]
# ### PLOTTING MAPS

# %%
from plotly.offline import init_notebook_mode
init_notebook_mode(connected = True)

# %%
import plotly
import plotly.graph_objs as go
import numpy as np
import math #needed for definition of pi


# %%
##Aggregate view of Non-Violent Crimes by State
data1 = dict(type='choropleth',
        colorscale = 'Viridis',
        autocolorscale = False,
        locations = data['state'],
        locationmode = 'USA-states',
        z = data['murders'].astype(float),
        colorbar = {'title':'murders (Per-100K-Pop)'}
        )
layout1 = dict(
        title = 'Aggregate view of non-Violent Crimes Per 100K Population',
        geo = dict(
            scope='usa',
            projection=dict( type='albers usa' ),
            showlakes = True,
            lakecolor='rgb(85,173,240)'),
             )
    
fig1 = go.Figure(data = [data1],layout = layout1)
plotly.offline.iplot(fig1,validate=False)
#go.iplot(fig1,validate=False)

# %% [markdown]
# #### Functions for Linear Regression

# %%
# add a constant to x to include the intercept in the regression
import statsmodels.api as sm

def linear_reg(x, y):
    x = sm.add_constant(x)
    # fit the regression model
    model = sm.OLS(y, x)
    results = model.fit()
    # print the regression results
    print(results.summary())
    prediction = results.predict(x)
    return results, prediction


# %%



