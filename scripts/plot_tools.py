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

def heatmeap_sns(ax : plt.Axes, data: pd.DataFrame, mask=True, vmax=1, vmin=-1):
    # Generate a mask for the upper triangle
    mask = np.triu(np.ones_like(data, dtype=bool))
    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(data, ax=ax, mask=mask, cmap=cmap, vmax=vmax, vmin=vmin, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})
    return ax

def barplot_pcat(ax, data: pd.DataFrame, xlevel:str, hue_vars:str ):
    dfm = data.reset_index(drop=False)
    list_colnames = []
    list_colnames.append(xlevel)
    list_colnames.extend(hue_vars)
    dfm = dfm.loc[:, list_colnames]
    dfm = dfm.melt(id_vars=xlevel)
    sns.barplot(data=dfm, x=xlevel, y='value', hue="variable",ax=ax)
    return ax

