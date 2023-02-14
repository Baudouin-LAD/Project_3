import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Optional


def addplotlabels(ax : plt.Axes, xlabel:str, ylabel:str) -> plt.Axes:
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    return ax

def boxplot(ax, data : pd.DataFrame,  xtitle : Optional[str] = None ):
#Boxplot of Violent crime variables
    sns.boxplot(data=data, ax=ax)
    return ax

def scatterplot(ax, data, xcol, ycol):
    ax.scatter(x=data[xcol],y=data[ycol])
    return ax

def histogram(ax,values, nbins=10):
    ax.hist(values, nbins=nbins)
    return ax

def pie_chart(data, categories, ycol=None):
    plot_data = data.loc[:,categories].reset_index(drop=False)
    plot_data= plot_data.T
    colnames = plot_data.iloc[0]
    plot_data.rename(columns=colnames,inplace=True)
    plot_data =plot_data.iloc[1:,:]
    axes = plot_data.plot.pie(subplots=True)
    return axes


def scatterplot_sns(ax : plt.Axes, data: pd.DataFrame, xcol:str, ycol:str, hue_col : Optional[str] = None):
    sns.scatterplot(data=data, x=xcol, y=ycol, hue=hue_col, ax=ax)
    return ax

def regplot_sns(ax : plt.Axes, data: pd.DataFrame, xcol:str, ycol:str):
    sns.regplot(data=data, x=xcol, y=ycol,  ax=ax)
    return ax

def heatmeap_sns(ax : plt.Axes, data: pd.DataFrame, mask=True, vmax=1, vmin=-1,center=0.5):
    # Generate a mask for the upper triangle
    #mask = np.triu(np.ones_like(data, dtype=bool))
    mask = np.triu(data)

    # Generate a custom diverging colormap
    #cmap = sns.diverging_palette(230, 20, as_cmap=True)
    cmap = sns.color_palette("Spectral_r", as_cmap=True)
    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(data, ax=ax, mask=mask,  vmax=vmax, vmin=vmin, center=center,
                cmap=cmap, annot=True,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})
    return ax

def corr_heatmap(data, vars_interest):
    corr_data = data.loc[:,vars_interest]
    corr_data = corr_data.corr()
    corr_data = np.abs(corr_data)

    corr_data.index = pd.CategoricalIndex(corr_data.index, 
                                       categories=vars_interest,
                                       )
    corr_data.sort_index(level=0, inplace=True)
    corr_data = corr_data[vars_interest]
    fig, ax = plt.subplots(1,1)
    heatmeap_sns(ax, corr_data, vmin=0, vmax=1)
    return corr_data, ax
    
    
def barplot_pcat(ax, data: pd.DataFrame, xlevel:str, hue_vars:str ):
    dfm = data.reset_index(drop=False)
    list_colnames = []
    list_colnames.append(xlevel)
    list_colnames.extend(hue_vars)
    dfm = dfm.loc[:, list_colnames]
    dfm = dfm.melt(id_vars=xlevel)
    sns.barplot(data=dfm, x=xlevel, y='value', hue="variable",ax=ax)
    return ax

def radar_chart(data, categories, states, title):
    # radar chart
    label_loc = np.linspace(start=0, stop=2 * np.pi, num=len(categories)+1)
    
    fig = plt.figure(figsize=(8, 8))
    plt.subplot(polar=True)
    for i, state in enumerate(states):
        state_df = data[data['state']==state]
        cat_p_pop = []
        for cate in categories:
            cat_p_pop.append((state_df[cate]/state_df["population"]).values[0])
            
        cat_p_pop = [*cat_p_pop, cat_p_pop[0]]
        plt.plot(label_loc, cat_p_pop, label=f'{state}')
    
    plt.title(title, size=20)
    categories = [*categories, categories[0]]
    lines, labels = plt.thetagrids(np.degrees(label_loc), labels=categories)
    plt.legend(loc = 'upper right')
    plt.show()
    return fig
#%%
import plotly
import plotly.graph_objs as go

import plotly.io as io
io.renderers.default='browser'

cmap = "Spectral_r"
def plotly_map_usa(data, variable, vmin, vmax,title :str,
                   colorbartitle :Optional[str] = None,
                   filename:Optional[str] = "fig.html",
                   image_width = 600,
                   image_height=600):
    ##Aggregate view of Non-Violent Crimes by State
    data1 = dict(type='choropleth',
            colorscale = cmap,
            autocolorscale = False,
            locations = data['state'],
            locationmode = 'USA-states',
            z = data[variable].astype(float),
            zmax=vmax,
            zmin=vmin,
            colorbar = {'title':f"{colorbartitle}"}
            )
    layout1 = dict(
            title = f"{title}",
            geo = dict(
                scope='usa',
                projection=dict( type='albers usa' ),
                showlakes = True,
                lakecolor='rgb(85,173,240)'),
                 )
        
    fig1 = go.Figure(data = [data1],layout = layout1)
    plotly.offline.plot(fig1,validate=False, image_width=image_width, image_height=image_height,  
                        filename=filename, auto_open=True)
    return fig1
