from tkinter import *
from tkinter import ttk
import customtkinter as ctk 
from PIL import Image, ImageTk
from ttkthemes import ThemedTk
from tkinter import messagebox
from tabulate import tabulate

import os
import pandasql as psql
import pandas as pd
pd.set_option('display.max_columns', None) 
pd.set_option('display.max_rows', None)
pd.set_option('display.expand_frame_repr', False)
import numpy as np
from scipy import stats
from scipy.stats import linregress,gaussian_kde,shapiro,ttest_ind
from sklearn.ensemble import IsolationForest
from sklearn.metrics import mean_squared_error

import json
import os
import re
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns

import traceback
import statsmodels.api as sm
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neighbors import LocalOutlierFactor


# general func
def get_dir(directory):
    try:
        contents = os.listdir(directory)
        for i in range(len(contents)):
            if os.path.isdir(os.path.join(directory, contents[i])):
                contents[i] += '/'
        return contents
    except FileNotFoundError:
        print(f"Directory '{directory}' not found.")
        return []
    except Exception as e:
        print(f"Error listing directory '{directory}': {e}")
        return []
def get_file_type(file_path:str):
        if file_path.endswith('/'):
            return 'dir'
        else:
            return file_path.split('.')[-1]    
def load_json(file_path):
    try:
        with open(file_path, 'r') as f:
            config = json.load(f)
        return config
    except FileNotFoundError:
        print("Configuration file not found.")
        return {}
    except json.JSONDecodeError:
        print("Error parsing configuration file.")
        return {}  
def read_data_file(file_full_path:str):
    file_type = file_full_path.split('.')[-1] 
    if file_type == 'csv':
        return pd.read_csv(file_full_path)
    elif file_type == 'xlsx':
        return pd.read_excel(file_full_path)    
    else:
        print('Unsupported file type.')
        return pd.DataFrame()  
def get_darker_color(hex_color, percentage=10):
    """
    Darkens the given hex color by the specified percentage.

    :param hex_color: str, the hex color string (e.g., "#E8DAEF").
    :param percentage: int, the percentage to darken the color (default is 10%).
    :return: str, the darkened hex color string.
    """
    # Ensure the input is a valid hex color
    try:
        if hex_color.startswith('#'):
            hex_color = hex_color[1:]

        # Convert hex to RGB
        r = int(hex_color[0:2], 16)
        g = int(hex_color[2:4], 16)
        b = int(hex_color[4:6], 16)

        # Darken the color by the specified percentage
        factor = (100 - percentage) / 100
        r = max(0, int(r * factor))
        g = max(0, int(g * factor))
        b = max(0, int(b * factor))

        # Convert RGB back to hex
        darkened_color = f"#{r:02x}{g:02x}{b:02x}"

        return darkened_color
    except:
        return 'black'    

# load config files
CURRENT_PATH = '\\'.join(os.path.abspath(__file__).split('\\')[:-1]) + '\\'
CONFIG = load_json(f'{CURRENT_PATH}config.json')

# charts
plt.rcParams['axes.facecolor'] = CONFIG['charts']['background']  # background
plt.rcParams['axes.edgecolor'] = get_darker_color(CONFIG['charts']['frame_color'],10) # frame & axes


# data frame func
def get_shape(df:pd.DataFrame,output_type:str='table'):
    data = pd.DataFrame(data=df.shape,columns=['#'],index=['rows','columns']).T
    data = data if output_type == 'table' else tabulate(data,headers='keys',tablefmt='psql') 
    return {
        'output':data,
        'output_type':output_type,
        'args':{'df':['df'],'output_type':[f"'table'",f"'text'"]}
        }  
def get_columns(df:pd.DataFrame):
    return {
        'output':pd.DataFrame(df.columns,columns=['Column']),
        'output_type':'table',
        'args':{'df':['df']}
        }  
def get_columns_info(df:pd.DataFrame,show='all',output_type:str='table'): 
    data = {'column':[],'type':[],'dtype':[],'unique':[],'Non-Nulls':[],'Non-Nulls%':[]}
    numeric_cols = df.select_dtypes(include=['number'])
    object_cols = df.select_dtypes(include=['object'])
    for column in df.columns:
        data['column'].append(column)
        data['type'].append('number' if column in numeric_cols else 'object')
        data['dtype'].append(str(df[column].dtype))
        data['unique'].append(len(df[column].unique().tolist()))
        data['Non-Nulls'].append(len(df[~df[column].isna()]))
        data['Non-Nulls%'].append(f"{round(len(df[~df[column].isna()])*100/len(df),2)}%")

    data = pd.DataFrame(data).sort_values(by=['type','dtype','Non-Nulls']).reset_index(drop=True)  

    if show != 'all':
        data = data[data.column==show].reset_index(drop=True)

    data = data if output_type == 'table' else tabulate(data,headers='keys',tablefmt='psql')
    return {
        'output':data,
        'output_type':output_type,
        'args':{'df':['df'],'show':["'all'"] + [f"'{col}'" for col in list(df.columns)],'output_type':[f"'table'",f"'text'"]}
        }  
def get_numerics_desc(df:pd.DataFrame,show='all',output_type:str='table'):
    data = df.describe().T
    numeric_columns = list(data.index)
    data['count'] = data['count'].astype(int)
    data["skewness"] = 3*(data['mean'] - data['50%'])/data['std']

    if show != 'all':
        data = data[data.index == show]

    data = data if output_type == 'table' else tabulate(data,headers='keys',tablefmt='psql')
    return {
        'output':data,
        'output_type':output_type,
        'args':{'df':['df'],'show':["'all'"] + [f'"{column}"' for column in numeric_columns],'output_type':[f"'table'",f"'text'"]}
        }
def get_categorical_desc(df:pd.DataFrame,show='all',outliers='None'):
    categorical_columns = [col for col in df.columns if str(df[col].dtype) in ['object','category','bool']]
    df = df[categorical_columns].copy()
    data = {'column':[],'type':[],'unique':[],'mode':[],'mode_occurances':[],'mode%':[],'prob_outliers':[],'outlier_items':[],'outliers_occurance_probability':[]}
    
    for column in df.columns:
        data['column'].append(column)
        data['type'].append(str(df[column].dtype))
        data['unique'].append(len(df[column].unique().tolist()))
        data['mode'].append(df[column].mode()[0])
        data['mode_occurances'].append(len(df[df[column]==df[column].mode()[0]]))
        data['mode%'].append(f"{100*len(df[df[column]==df[column].mode()[0]])/len(df):.2f}%")

        if outliers in [None,'None','none']:
            data['prob_outliers'].append([])
            data['outlier_items'].append([])
            data['outliers_occurance_probability'].append([])
        else:    
            PERCENTAGE = float(outliers[:outliers.find('%')])
            df['occ_prob'] = df[column].map(df[column].value_counts(normalize=True))
            data['prob_outliers'].append(len(df[df['occ_prob'] < PERCENTAGE/100]))
            data['outlier_items'].append(str(list(df.loc[df['occ_prob'] < PERCENTAGE/100,column].unique())))
            data['outliers_occurance_probability'].append(str(list(df.loc[df['occ_prob'] < PERCENTAGE/100,'occ_prob'].unique())))
        
    data = pd.DataFrame(data).reset_index(drop=True)

    if show != 'all':
        data = data[data['column'] == show]

    return {
        'output':data,
        'output_type':'table',
        'args':{'df':['df'],'show':["'all'"] + [f'"{column}"' for column in categorical_columns],'outliers':[f'"None"',f'"0.3%"',f'"0.5%"',f'"1%"',f'"5%"',f'"10%"']}
        }
def get_preview(df:pd.DataFrame,rows=5,end='head',output_type:str='table'):
    
    if rows >= len(df):
        data = df 
    elif end == 'head':
        data = df.head(rows) 
    elif end == 'tail':
        data = df.tail(rows) 
    elif end == 'random':
        data = df.sample(rows)     
    else:
        data = pd.DataFrame() 

    data = data if output_type == 'table' else tabulate(data,headers='keys',tablefmt='psql')  

    return {
        'output':data,
        'output_type':output_type,
        'args':{'df':['df'],'rows':[3,5,10,25],'end':[f"'head'",f"'tail'",f"'random'"],'output_type':[f"'table'",f"'text'"]}
        }
def get_data(df:pd.DataFrame,sql:str,output_type:str='table'):
    try:
        data = psql.sqldf(sql)
        data = data if output_type == 'table' else tabulate(data,headers='keys',tablefmt='psql')
    except Exception as e:
        output_type = 'text' 
        data = f"\n>>> Error processing SQL:\n{e}\n"
           
    #print(f" >>> output:{data}\n >>> output_type:{output_type}") # monitor
    return {
        'output':data,
        'output_type':output_type,
        'args':{'df':['df'],'sql':[f"'SELECT * FROM df LIMIT 10'"],'output_type':[f"'table'",f"'text'"]}
        }

def get_correlations(df:pd.DataFrame,in_chart:bool=False):

    data = df[df.select_dtypes(include=['number']).columns].corr()
    abs_data = data.abs()
    np.fill_diagonal(abs_data.values, np.nan)
    corr_list = abs_data.unstack()
    corr_list = corr_list.dropna().sort_values(ascending=False)
    corr_df = corr_list.reset_index()
    corr_df.columns = ['column1', 'column2', 'r^2']
    corr_df = corr_df.drop_duplicates(subset='r^2').reset_index(drop=True)

    if in_chart:
        NUM_OF_COLUMNS = len(data.columns)
        SIZE = NUM_OF_COLUMNS if NUM_OF_COLUMNS < 12 else int(0.5*NUM_OF_COLUMNS)
        fig, ax = plt.subplots(figsize=(SIZE,SIZE),dpi=75)
        colors = [CONFIG['charts']['data_colors'][1],CONFIG['charts']['data_colors'][0],CONFIG['charts']['data_colors'][1]] # green,purple,green
        cmap = LinearSegmentedColormap.from_list('custom_diverging', colors, N=100)

        im = ax.imshow(data, cmap=cmap, interpolation='nearest', vmin=-1, vmax=1)
        fig.colorbar(im, ax=ax, fraction=0.05)
        ax.set_xticks(range(len(data.columns)))
        ax.set_yticks(range(len(data.columns)))
        ax.set_xticklabels(data.columns, rotation=45, ha='right')
        ax.set_yticklabels(data.columns)

        if NUM_OF_COLUMNS < 36:
            for i in range(len(data.columns)):
                for j in range(len(data.columns)):
                    text = ax.text(j, i, f'{data.iloc[i, j]:.2f}',
                                ha='center', va='center', color='black')

        plt.tight_layout()           

    return {
        'output':data if in_chart==False else fig,
        'output_type':'chart' if in_chart==True else 'table',
        'title':"'r^2' Correlations Heatmap:",
        'table':corr_df,
        'args':{'df':['df'],'in_chart':[True,False]}
        }
def get_correlation_plot(df:pd.DataFrame,y:str='None',x:str='None',by:str='None',reg_type='linear',exclude_outliers="None",show_outliers='True'):
    if x in [None,'None','none']:
        x = list(df.select_dtypes(include=['number']).columns)[0] 
    if y in [None,'None','none']:
        y = list(df.select_dtypes(include=['number']).columns)[1]     

    data = df[[x,y]].dropna().copy() if by in [None,"None",'none'] else df[[x,y,by]].dropna().copy()
    data['inlier'] = 1

    POINT_SIZE = 4 if len(data) > 1000 else 8 if len(data) > 200 else 11
    ALPHA = 0.5 if len(data) > 1000 else 0.7 if len(data) > 200 else 0.8

    fig, axs = plt.subplots(
        2,2,
        figsize=(5,5),
        dpi=80,
        sharex='col', sharey='row',
        gridspec_kw={'height_ratios': [1,6],'width_ratios': [6,1]}
        )
      
    plt.subplots_adjust(wspace=0,hspace=0) 
    fig.delaxes(axs[0,1])
    axs[1,0].set_xlabel(x)
    axs[1,0].set_ylabel(y)
    axs[0, 0].axis('off')
    axs[1, 1].axis('off')
    axs[0, 0].tick_params(axis="x", labelbottom=False)
    axs[1, 1].tick_params(axis="y", labelleft=False)

    for side in ['top','bottom','right','left']: 
        axs[1,0].spines[side].set_linewidth(1)

    if by in [None,"None",'none']: # no categories chart
    
        summary_table = {
            'category':['all'],
            'count':[len(data)],
            'type':[reg_type],
            'pred_func':[],
            'r^2':[],
            'rmse':[],
            'std_err':[],
            'excluded_outliers':[]
        }

        COLOR_INDEX = CONFIG['charts']['data_colors'][0]
        HIST_BINS = min(50,data.shape[0])

        if exclude_outliers not in ['None','none',None]:
            PERCENTAGE = 0.25 if exclude_outliers == 'IQR' else float(exclude_outliers[:exclude_outliers.find('%')])/100
            iso_forest = IsolationForest(n_estimators=100, contamination=PERCENTAGE, random_state=42)
            data['inlier'] = iso_forest.fit_predict(data)
            summary_table['excluded_outliers'].append(len(data[data.inlier==-1]))

            if show_outliers in ['True','true',True]:
                axs[1,0].plot(data.loc[data.inlier==-1,x],data.loc[data.inlier==-1,y], # outliers dp
                    linestyle='none', 
                    linewidth=1.5, 
                    alpha=0.9,
                    marker='o', 
                    markersize=POINT_SIZE, 
                    markerfacecolor= CONFIG['charts']['data_colors'][0],
                    markeredgecolor='red'
                )
                axs[0, 0].hist(data[x], alpha=ALPHA-0.2, bins=HIST_BINS, edgecolor=get_darker_color(COLOR_INDEX,70), color=COLOR_INDEX)
                axs[1, 1].hist(data[y], alpha=ALPHA-0.2, bins=HIST_BINS,orientation='horizontal',edgecolor=get_darker_color(COLOR_INDEX,70), color=COLOR_INDEX)
            else:
                axs[0, 0].hist(data.loc[data.inlier==1,x], alpha=ALPHA-0.2, bins=HIST_BINS, edgecolor=get_darker_color(COLOR_INDEX,70), color=COLOR_INDEX)
                axs[1, 1].hist(data.loc[data.inlier==1,y], alpha=ALPHA-0.2, bins=HIST_BINS,orientation='horizontal',edgecolor=get_darker_color(COLOR_INDEX,70), color=COLOR_INDEX)    
        else:
            axs[0, 0].hist(data[x], alpha=ALPHA-0.2, bins=HIST_BINS, edgecolor=get_darker_color(COLOR_INDEX,70), color=COLOR_INDEX)
            axs[1, 1].hist(data[y], alpha=ALPHA-0.2, bins=HIST_BINS,orientation='horizontal',edgecolor=get_darker_color(COLOR_INDEX,70), color=COLOR_INDEX)
            summary_table['excluded_outliers'].append(0)    

        axs[1,0].plot(data.loc[data.inlier==1,x],data.loc[data.inlier==1,y], # normal dp
                linestyle='none', 
                linewidth=1, 
                alpha=ALPHA,
                marker='o', 
                markersize=POINT_SIZE, 
                markerfacecolor= CONFIG['charts']['data_colors'][0],
                markeredgecolor=get_darker_color(CONFIG['charts']['data_colors'][0],50)
            )    

        if reg_type == 'linear':
            slope, intercept, r_value, p_value, std_err = linregress(data.loc[data.inlier==1,x], data.loc[data.inlier==1,y])
            regression_line = slope * data.loc[data.inlier==1,x] + intercept
            axs[1,0].plot(data.loc[data.inlier==1,x], regression_line, color='red', label=f'Linear Fit: $y={slope:.2f}x+{intercept:.2f}$')
            summary_table['r^2'].append(f"{r_value**2:.4f}")
            summary_table['std_err'].append(std_err)
            summary_table['rmse'].append(np.sqrt(mean_squared_error(data.loc[data.inlier==1,y],regression_line)))
            summary_table['pred_func'].append(f"y={slope:.4f}x+{intercept:.4f}")
        else:
            summary_table['r^2'].append(None)
            summary_table['std_err'].append(None)
            summary_table['rmse'].append(None)
            summary_table['pred_func'].append(None)    
    
    else: # chart by categories
        summary_table = {
            'category':[],
            'count':[],
            'type':[],
            'pred_func':[],
            'r^2':[],
            'rmse':[],
            'std_err':[],
            'excluded_outliers':[]
            }

        for i,category in enumerate(data[by].unique()):
            
            summary_table['type'].append(reg_type)
            summary_table['category'].append(category)
            summary_table['count'].append(len(data[data[by]==category]))
            COLOR_INDEX =  i % len(CONFIG['charts']['data_colors'])
            HIST_BINS = min(50,data.loc[data[by]==category,x].shape[0])

            if exclude_outliers not in ['None',None,'none']:
                PERCENTAGE = 0.25 if exclude_outliers == 'IQR' else float(exclude_outliers[:exclude_outliers.find('%')])/100
                iso_forest = IsolationForest(n_estimators=100, contamination=PERCENTAGE, random_state=42)
                data['inlier'] = iso_forest.fit_predict(data[[x,y]])
                summary_table['excluded_outliers'].append(len(data[(data.inlier==-1)&(data[by]==category)]))

                if show_outliers in ['True','true',True]:
                    axs[1,0].plot(data.loc[(data[by]==category)&(data.inlier==-1),x],data.loc[(data[by]==category)&(data.inlier==-1),y], # outliers dp
                    linestyle='none', 
                    linewidth=1.5, 
                    alpha=0.9,
                    marker='o', 
                    markersize=POINT_SIZE, 
                    markerfacecolor=CONFIG['charts']['data_colors'][COLOR_INDEX],
                    markeredgecolor='red'
                    )

                    axs[0, 0].hist(data.loc[data[by]==category,x], bins=HIST_BINS, alpha=ALPHA-0.2,edgecolor=get_darker_color(CONFIG['charts']['data_colors'][COLOR_INDEX],70), color=CONFIG['charts']['data_colors'][COLOR_INDEX])
                    axs[1, 1].hist(data.loc[data[by]==category,y], bins=HIST_BINS, alpha=ALPHA-0.2,orientation='horizontal', edgecolor=get_darker_color(CONFIG['charts']['data_colors'][COLOR_INDEX],70), color=CONFIG['charts']['data_colors'][COLOR_INDEX])
                else:
                    axs[0, 0].hist(data.loc[(data[by]==category)&(data.inlier==1),x], bins=HIST_BINS, alpha=ALPHA-0.2,edgecolor=get_darker_color(CONFIG['charts']['data_colors'][COLOR_INDEX],70), color=CONFIG['charts']['data_colors'][COLOR_INDEX])
                    axs[1, 1].hist(data.loc[(data[by]==category)&(data.inlier==1),y], bins=HIST_BINS, alpha=ALPHA-0.2,orientation='horizontal', edgecolor=get_darker_color(CONFIG['charts']['data_colors'][COLOR_INDEX],70), color=CONFIG['charts']['data_colors'][COLOR_INDEX])         
            else:
                axs[0, 0].hist(data.loc[data[by]==category,x], bins=HIST_BINS, alpha=ALPHA-0.2,edgecolor=get_darker_color(CONFIG['charts']['data_colors'][COLOR_INDEX],70), color=CONFIG['charts']['data_colors'][COLOR_INDEX])
                axs[1, 1].hist(data.loc[data[by]==category,y], bins=HIST_BINS, alpha=ALPHA-0.2,orientation='horizontal', edgecolor=get_darker_color(CONFIG['charts']['data_colors'][COLOR_INDEX],70), color=CONFIG['charts']['data_colors'][COLOR_INDEX])
                summary_table['excluded_outliers'].append(0)

            axs[1,0].plot(data.loc[(data[by]==category) & (data.inlier==1),x],data.loc[(data[by]==category) & (data.inlier==1),y], # normal dp
                linestyle='none', 
                linewidth=1, 
                alpha=ALPHA,
                marker='o', 
                markersize=POINT_SIZE, 
                markerfacecolor=CONFIG['charts']['data_colors'][COLOR_INDEX], 
                markeredgecolor=get_darker_color(CONFIG['charts']['data_colors'][COLOR_INDEX],50)
            )

            if reg_type == 'linear':
                slope, intercept, r_value, p_value, std_err = linregress(data.loc[(data[by]==category)&(data.inlier==1),x],data.loc[(data[by]==category)&(data.inlier==1),y])
                regression_line = slope * data.loc[(data[by]==category)&(data.inlier==1),x] + intercept
                axs[1,0].plot(data.loc[(data[by]==category)&(data.inlier==1),x], regression_line, color=get_darker_color(CONFIG['charts']['data_colors'][COLOR_INDEX],30), label=f'Linear Fit: $y={slope:.2f}x+{intercept:.2f}$')
                summary_table['r^2'].append(f"{r_value**2:.4f}")
                summary_table['rmse'].append(np.sqrt(mean_squared_error(data.loc[(data[by]==category)&(data.inlier==1),y],regression_line)))
                summary_table['std_err'].append(std_err)
                summary_table['pred_func'].append(f"y={slope:.4f}x+{intercept:.4f}")
            else:
                summary_table['r^2'].append(None)
                summary_table['rmse'].append(None)
                summary_table['std_err'].append(None)
                summary_table['pred_func'].append(None)    

    try: # monitor
        sum_table = pd.DataFrame(summary_table) 
        #print(summary_table)
    except Exception as e:
        print(e)
        sum_table = pd.DataFrame({'Error occured creating summary table':e},index=[0])
            
    plt.tight_layout() 
    return {
        'output':fig,
        'output_type':'chart',
        'title':f' y="{y}"(=Numeric) by X="{x}"(=Numeric)' if by in [None,'none','None'] else f' y="{y}"(=Numeric) by X="{x}"(=Numeric) by "{by}"(=Category)' ,
        'table':sum_table,
        'args':{'df':['df'],
                'x':[f'"{item}"' for item in list(df.select_dtypes(include=['number']).columns)],
                'y':[f'"{item}"' for item in list(df.select_dtypes(include=['number']).columns)],
                'by':["None"] + [f'"{item}"' for item in list(df.select_dtypes(include=['object']).columns) if len(df[item].unique()) < 16],
                'reg_type':["None",f'"linear"'],
                'exclude_outliers':['"None"',"'IQR'",'"0.3%"','"0.5%"','"1%"','"5%"'],
                'show_outliers':[f"'True'",f"'False'"]}
        }
def get_dist_plot(df:pd.DataFrame,x:str=None,by:str=None,outliers="none"):
    def get_stats(data):
        return {
            'count':len(data),
            'min':data.min(),
            'max':data.max(),
            'mean':data.mean(),
            'median':data.median(),
            'skewness':(3*(data.mean() - data.median()))/data.std() if data.std() != 0 else 0,
            'std':data.std(),
            'q1':data.quantile(0.25),
            'q3':data.quantile(0.75),
            'lcl':data.quantile(0.003),
            '-3*std':data.quantile(0.003),
            'ucl':data.quantile(0.997),
            '+3*std':data.quantile(0.997),
            'iqr':data.quantile(0.75) - data.quantile(0.25),
            'IQR':f"[{data.quantile(0.25)}:{data.quantile(0.75)}]",
            'lower_whisker':max(data.min(), data.quantile(0.25) - 1.5 * (data.quantile(0.75) - data.quantile(0.25))),
            'upper_whisker':min(data.max(), data.quantile(0.75) + 1.5 * (data.quantile(0.75) - data.quantile(0.25))),
            'outliers':[]
        }
    def set_vlines(ax,stats:{},keys:{}):
        for key,color in keys.items():
            #print(f"{key}:{stats[key]}(color={color})")
            LABEL,VALUE,COLOR = key,stats[key],color
            ax.axvline(VALUE, color=COLOR, linestyle='-',linewidth=2) #label=f'{LABEL} = {VALUE:.2f}'
            ax.text(VALUE, 0, LABEL, horizontalalignment="center", verticalalignment="top", transform=ax.get_xaxis_transform(), rotation=45,color=COLOR)           
    def set_data(data:pd.DataFrame,x:str,outliers=None):
        #inliers_data,outliers_data = pd.DataFrame(columns=data.columns), pd.DataFrame(columns=data.columns)
        STATS = get_stats(data[x])

        if any([item in outliers for item in ['iqr','IQR']]):
            #print(' >>> iqr')
            outliers_data = data.loc[(data[x] < STATS['lower_whisker'])|(data[x] > STATS['upper_whisker']),:].copy()
            inliers_data = data.loc[(data[x] >= STATS['lower_whisker'])&(data[x] <= STATS['upper_whisker']),:].copy()
        elif '%' in outliers:     
            #print(' >>> %')
            perc_string = outliers.split('_')[1]
            PERCENTAGE = float(perc_string[:perc_string.find('%')])
            #print(f" >>> PERCENTAGE={PERCENTAGE}")
            iso_forest = IsolationForest(n_estimators=200, contamination=PERCENTAGE/100, random_state=42)
            data['inlier'] = iso_forest.fit_predict(data[[x]])
            outliers_data = data.loc[data['inlier']==-1,:].drop('inlier', axis=1).copy()
            inliers_data = data.loc[data['inlier']==1,:].drop('inlier', axis=1).copy()
        else: # outliers in [None,"None",'none']
            #print(' >>> no outliers')    
            inliers_data = data.loc[:,:].copy()
            outliers_data = data.loc[0:-1,:].copy()

        #print(f" >>> inliers_data={len(inliers_data)}\n >>> outliers_data={len(outliers_data)}")
        return inliers_data,outliers_data
    def set_kde(ax,data,color='#d89fee'):
        kde_x = np.linspace(data.values.min(), data.values.max(),100)
        kde_y = gaussian_kde(data.values)(np.linspace(data.values.min(), data.values.max(),100))
        ax.twinx().plot(kde_x,kde_y, color=get_darker_color(color,30), label='Density', linewidth=2)   

    if x in [None,'none','None']:
        x = list(df.select_dtypes(include=['number']).columns)[0]  

    data= df[[x]].dropna().reset_index(drop=True) if by in ['none','None',None] else df[[x,by]].dropna().reset_index(drop=True)
    inliers_data,outliers_data = set_data(data=data,x=x,outliers=outliers)
    STATS = get_stats(data[x])    
    st = { # summary table
            'category':['all'],
            'count':[f"{STATS['count']:.2f}"],
            'min':[f"{STATS['min']:.2f}"],
            'mean':[f"{STATS['mean']:.2f}"],
            'median':[f"{STATS['median']:.4f}"],
            'std':[f"{STATS['std']:.2f}"],
            'max':[f"{STATS['max']:.2f}"],
            'IQR':[f"[{STATS['q1']:.2f}:{STATS['q3']:.2f}]"],
            'skewness':[f"{STATS['skewness']:.2f}"],
            'outliers':[len(outliers_data)]
            }

    if by in [None,'none','None']:
        POINT_SIZE = 5 if len(inliers_data) > 1000 else 8 if len(inliers_data) > 200 else 9
        ALPHA = 0.1 if len(inliers_data) > 1000 else 0.4 if len(inliers_data) > 200 else 0.6
        HEIGHT = 3
        fig, axs = plt.subplots(2,1,figsize=(6,HEIGHT),dpi=75,sharex=True,gridspec_kw={'height_ratios': [HEIGHT,3]})

        sns.stripplot(inliers_data[x],ax=axs[0],orient='h',alpha=ALPHA,size=POINT_SIZE,linewidth=0.5,color=CONFIG['charts']['data_colors'][0],edgecolor=get_darker_color(CONFIG['charts']['data_colors'][0],70),jitter=0.35,zorder=0) 
        if any([item in outliers for item in ['exclude','Exclude','EXCLUDE']]):
            STATS = get_stats(inliers_data[x])
            sns.boxplot(inliers_data[x],orient="h",linewidth=2,boxprops={"facecolor": "none", "edgecolor": "black", "linewidth": 1.5},showfliers=False, ax=axs[0])
            axs[1].hist(inliers_data[x],bins=min(len(inliers_data[x]),50),color=CONFIG['charts']['data_colors'][0],edgecolor=get_darker_color(CONFIG['charts']['data_colors'][0],70), alpha=0.3)
            if len(inliers_data) > 100:
                set_kde(ax=axs[1],data=inliers_data[x],color=CONFIG['charts']['data_colors'][0])
        else:
            STATS = get_stats(data[x])
            sns.stripplot(outliers_data[x],ax=axs[0],orient='h',alpha=0.4,size=POINT_SIZE,linewidth=0.5,color='red',edgecolor=get_darker_color(CONFIG['charts']['frame_color'],70),jitter=0.3,zorder=0)   
            sns.boxplot(data[x],orient="h",linewidth=2,boxprops={"facecolor": "none", "edgecolor": "black", "linewidth": 1.5},showfliers=False, ax=axs[0])    
            axs[1].hist(data[x],bins=min(len(data[x]),50),color=CONFIG['charts']['data_colors'][0],edgecolor=get_darker_color(CONFIG['charts']['data_colors'][0],70), alpha=0.3)
            if len(data[x]) > 100:
                    set_kde(ax=axs[1],data=data[x],color=CONFIG['charts']['data_colors'][0])

    else: # by categories
        POINT_SIZE = 5 if len(data) > 1000 else 8 if len(data) > 200 else 9
        HEIGHT = int(1.5*len(data[by].unique()))
        fig, axs = plt.subplots(2,1,figsize=(6,HEIGHT),dpi=75,sharex=True,gridspec_kw={'height_ratios': [HEIGHT,3]})

        if any([item in outliers for item in ['exclude','Exclude','EXCLUDE']]):
            sns.boxplot(x=inliers_data[x],y=inliers_data[by],orient="h",linewidth=2,boxprops={"facecolor": "none", "edgecolor": "black", "linewidth": 1.5},showfliers=False, ax=axs[0])
        else:
            sns.boxplot(x=data[x],y=data[by],orient="h",linewidth=2,boxprops={"facecolor": "none", "edgecolor": "black", "linewidth": 1.5},showfliers=False, ax=axs[0])

        for i,cat in enumerate(data[by].unique()):
            
            COLOR_INDEX = i % len(CONFIG['charts']['data_colors'])
            ALPHA = 0.1 if len(data[data[by]==cat]) > 1000 else 0.4 if len(data[data[by]==cat]) > 200 else 0.6
            inliers_data,outliers_data = set_data(data=data.loc[data[by]==cat],x=x,outliers=outliers)
            print(f" >>> category={cat}, inliers={len(inliers_data)}, outliers={len(outliers_data)}")

            # update summary table
            STATS = get_stats(data.loc[data[by]==cat,x]) 
            for key in st.keys():
                st[key].append(len(outliers_data) if key=='outliers' else cat if key=='category' else STATS[key])

            sns.stripplot(x=inliers_data.loc[inliers_data[by]==cat,x],y=[cat]*len(inliers_data.loc[inliers_data[by]==cat,x]),ax=axs[0],orient='h',alpha=ALPHA,size=POINT_SIZE,linewidth=0.5,color=CONFIG['charts']['data_colors'][COLOR_INDEX],edgecolor=get_darker_color(CONFIG['charts']['data_colors'][COLOR_INDEX],70),jitter=0.35,zorder=0) 
            if any([item in outliers for item in ['exclude','Exclude','EXCLUDE']]):
                STATS = get_stats(inliers_data[x])
                axs[1].hist(inliers_data.loc[inliers_data[by]==cat,x],bins=min(len(inliers_data.loc[inliers_data[by]==cat,x]),50),color=CONFIG['charts']['data_colors'][COLOR_INDEX],edgecolor=get_darker_color(CONFIG['charts']['data_colors'][COLOR_INDEX],70), alpha=0.3)
                if len(inliers_data.loc[inliers_data[by]==cat,x]) > 100:
                    set_kde(ax=axs[1],data=inliers_data.loc[inliers_data[by]==cat,x],color=CONFIG['charts']['data_colors'][COLOR_INDEX])
            else:    
                sns.stripplot(x=outliers_data[x],y=[cat]*len(outliers_data.loc[outliers_data[by]==cat,:]),ax=axs[0],orient='h',alpha=0.4,size=POINT_SIZE,linewidth=0.5,color='red',edgecolor=get_darker_color(CONFIG['charts']['frame_color'],70),jitter=0.3,zorder=0)   
                axs[1].hist(data.loc[data[by]==cat,x],bins=min(len(data.loc[data[by]==cat,x]),50),color=CONFIG['charts']['data_colors'][COLOR_INDEX],edgecolor=get_darker_color(CONFIG['charts']['data_colors'][COLOR_INDEX],70), alpha=0.3)
                if len(data.loc[data[by]==cat,x]) > 100:
                    set_kde(ax=axs[1],data=data.loc[data[by]==cat,x],color=CONFIG['charts']['data_colors'][COLOR_INDEX])

            #set_vlines(ax=axs[1],stats=STATS,keys={'mean':get_darker_color(CONFIG['charts']['data_colors'][COLOR_INDEX],60),'median':get_darker_color(CONFIG['charts']['data_colors'][COLOR_INDEX],30)})   
    
    for side in ['top','bottom','right','left']: 
        axs[0].spines[side].set_linewidth(1)
        axs[1].spines[side].set_linewidth(1)
    
    plt.tight_layout()
    axs[0].set(xlabel=None)
    axs[0].set(ylabel=None)
    axs[1].set_ylabel("Count")
    axs[1].twinx().set_ylabel("Density")
    axs[1].legend()

    return {
        'output':fig,
        'output_type':'chart',
        'title':f'"{x}" Values distribution:',
        'table':pd.DataFrame(st),
        'args':{
            'df':['df'],
            'x':[f'"{item}"' for item in list(df.select_dtypes(include=['number']).columns)],
            'by':["None"] + [f'"{item}"' for item in list(df.select_dtypes(include=['object']).columns) if len(df[item].unique()) < 16],
            'outliers':[f'"None"',f'"show_IQR"',f'"show_0.3%"',f'"show_0.5%"',f'"show_1%"',f'"show_5%"',f'"exclude_IQR"',f'"exclude_0.3%"',f'"exclude_0.5%"',f'"exclude_1%"',f'"exclude_5%"'],
            'exclude_outliers':["False","True"]
            }
        }

def get_compare_plot(df:pd.DataFrame,y='None',category='None',alpha:float=0.05,show_outliers='None'):
    def get_stats(data):
        return {
            'count':len(data),
            'min':data.min(),
            'max':data.max(),
            'mean':data.mean(),
            'median':data.median(),
            'skewness':data.mean() - data.median(),
            'std':data.std(),
            'q1':data.quantile(0.25),
            'q3':data.quantile(0.75),
            'lcl':data.quantile(0.003),
            '-3*std':data.quantile(0.003),
            'ucl':data.quantile(0.997),
            '+3*std':data.quantile(0.997),
            'iqr':data.quantile(0.75) - data.quantile(0.25),
            'lower_whisker':max(data.min(), data.quantile(0.25) - 1.5 * (data.quantile(0.75) - data.quantile(0.25))),
            'upper_whisker':min(data.max(), data.quantile(0.75) + 1.5 * (data.quantile(0.75) - data.quantile(0.25)))
        }
    
    MAX_CATEGORIES,SIGNIFICANCE = 30,alpha

    if category in ['None',None]:
        category = [item for item in list(df.select_dtypes(include=['object']).columns) if len(df[item].unique()) < MAX_CATEGORIES][0]
    if y in ['None',None]:
        y = list(df.select_dtypes(include=['number']).columns)[0]  

    data = df[[category,y]].dropna().copy()
    WIDTH,HEIGHT = min(5 + len(data[category].unique())*1 ,6),5 # len(data[category].unique()),5
    fig, ax = plt.subplots(figsize=(WIDTH,HEIGHT),dpi=75)

    st = {'category':[],'count':[],'min':[],'mean':[],'median':[],'std':[],'max':[],'outliers':[],'t-test_p_value':[],'decision':[]}
    sns.boxplot(data=data,x=category,y=y,color="white", linewidth=2, showfliers=False,ax = ax,zorder=100)
    
    for i,cat in enumerate(data[category].unique()):
        
        dp_y = data.loc[data[category]==cat,y]
        dp_y_others = data.loc[data[category]!=cat,y]
        dp_x = data.loc[data[category]==cat,category]

        t_statistic, p_value = ttest_ind(dp_y, dp_y_others) # t_test for independent groups 
        st['t-test_p_value'].append(p_value)
        st['decision'].append("significant" if p_value < SIGNIFICANCE else 'inSignificant')
        st['category'].append(cat)
        st['count'].append(len(dp_y))
        st['min'].append(min(dp_y))
        st['mean'].append(np.mean(dp_y))
        st['median'].append(np.median(dp_y))
        st['std'].append(np.std(dp_y))
        st['max'].append(max(dp_y))

        POINT_SIZE = 4 if len(dp_y) > 1000 else 6 if len(dp_y) > 200 else 8
        ALPHA = 0.3 if len(dp_y) > 1000 else 0.4 if len(dp_y) > 200 else 0.5
        DP_COLOR = CONFIG['charts']['data_colors'][i % len(CONFIG['charts']['data_colors'])] 

        if '%' in show_outliers:
            PERCENTAGE = float(show_outliers[:show_outliers.find('%')])
            LCL = dp_y.quantile((PERCENTAGE/2)/100)
            UCL = dp_y.quantile(1 - ((PERCENTAGE/2)/100))
            dp_ouliers = dp_y[(dp_y > UCL)|(dp_y < LCL)] 
            dp_y = dp_y[(dp_y <= UCL) & (dp_y >= LCL)] 
            st['outliers'].append(len(dp_ouliers))
            sns.stripplot(y=dp_ouliers,x=len(dp_ouliers)*[cat],alpha=0.9,size=POINT_SIZE,linewidth=1,color=DP_COLOR,edgecolor='red',jitter=0.35) 
            sns.stripplot(y=dp_y,x=len(dp_y)*[cat],alpha=ALPHA,size=POINT_SIZE,linewidth=0.5,color=DP_COLOR,edgecolor=CONFIG['charts']['frame_color'],jitter=0.35) 
        elif show_outliers == 'IQR':
            STATS = get_stats(dp_y)
            dp_ouliers = dp_y[(dp_y > STATS['upper_whisker'])|(dp_y < STATS['lower_whisker'])] 
            dp_y = dp_y[(dp_y <= STATS['upper_whisker']) & (dp_y >= STATS['lower_whisker'])] 
            st['outliers'].append(len(dp_ouliers))
            sns.stripplot(y=dp_ouliers,x=len(dp_ouliers)*[cat],alpha=0.9,size=POINT_SIZE,linewidth=1,color=DP_COLOR,edgecolor='red',jitter=0.35) 
            sns.stripplot(y=dp_y,x=len(dp_y)*[cat],alpha=ALPHA,size=POINT_SIZE,linewidth=0.5,color=DP_COLOR,edgecolor=CONFIG['charts']['frame_color'],jitter=0.35)     
        else:
            sns.stripplot(y=dp_y,x=dp_x,alpha=ALPHA,size=POINT_SIZE,linewidth=0.5,color=DP_COLOR,edgecolor=CONFIG['charts']['frame_color'],jitter=0.35)  
            st['outliers'].append('None') 
        
        if len(dp_y) > 1:
            sns.stripplot(y=[np.mean(dp_y)],x=[cat],color='cyan',size=POINT_SIZE,marker='D',edgecolor=CONFIG['charts']['frame_color'])

        if len(data[category].unique()) > 8:
            ax.tick_params(axis='x', rotation=45)

        sns.lineplot(x=st['category'],y=st['mean'],color='blue',linewidth=1,zorder=100)    
    
    return {
        'output':fig,
        'output_type':'chart',
        'title':f"Compare '{y}'(=Numeric) by '{category}' (=Category):\n",
        'table':pd.DataFrame(st),
        'args':{
            'df':['df'],
            'y':[f'"{item}"' for item in list(df.select_dtypes(include=['number']).columns)],
            'category':[f'"{item}"' for item in list(df.select_dtypes(include=['object']).columns) if len(df[item].unique()) < MAX_CATEGORIES],
            'alpha':[0.05,0.1,0.01],
            'show_outliers':[f"'None'",f"'IQR'",f"'0.3%'",f"'0.5%'",f"'1%'",f"'5%'",f"'10%'"]
            }
        }
def get_category_compare(df:pd.DataFrame,y:str='None',category:str='None',alpha:float=0.05):
    
    def plot_chi2_distribution(ax, chi2, dof, alpha=0.05):
        x = np.linspace(0, chi2*2, 1000)
        y = stats.chi2.pdf(x, dof)
        critical_value = stats.chi2.ppf(1 - alpha, dof)
        
        ax.plot(x, y, 'b-', lw=2, label='Chi-Square Distribution',color=CONFIG['charts']['data_colors'][0])
        ax.fill_between(x[x > critical_value], y[x > critical_value], color='green', alpha=0.3, label='Critical Region')
        ax.axvline(chi2, color=CONFIG['charts']['data_colors'][0], linestyle='--', label=f'Result: {chi2:.2f}')
        ax.axvline(critical_value, color='green', linestyle=':', label=f'Critical Value: {critical_value:.2f}')
        ax.set_title(f'Chi-Square Distribution (df={dof}, α={alpha})')
        ax.set_xlabel('Chi-Square Value')
        ax.set_ylabel('Probability Density')
        ax.legend()
        
        # Add text to show if chi2 exceeds critical value
        if chi2 > critical_value:
            ax.text(0.05, 0.95, 'Chi-square statistic exceeds critical value\nReject null hypothesis', 
                    transform=ax.transAxes, verticalalignment='top', 
                    bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
        else:
            ax.text(0.05, 0.95, 'Chi-square statistic does not exceed critical value\nFail to reject null hypothesis', 
                    transform=ax.transAxes, verticalalignment='top', 
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    def plot_cell_contributions(ax, observed, expected):
        contributions = (observed - expected)**2 / expected
        categories = [f'X^2({i} & {j})' for i in observed.index for j in observed.columns]
        bars = ax.bar(categories, contributions.values.flatten(),color=CONFIG['charts']['data_colors'][0],edgecolor=get_darker_color(CONFIG['charts']['data_colors'][0],30),linewidth=1)
        ax.set_title('Category Contributions to Chi-Square Statistic')
        ax.set_xlabel('Cells (row,column)')
        ax.set_ylabel('X^2[=(obs-exp)^2/exp] Contribution')
        ROTATION = 15
        ax.tick_params(axis='x', rotation=ROTATION)
    
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,f'{height:.2f}',ha='center', va='bottom', rotation=15,color=CONFIG['charts']['font_color'])
    def plot_percentage_comparison(ax, observed, expected):
        observed_flat = observed.values.flatten()
        expected_flat = expected.flatten()
        categories = [f'{i},{j}' for i in observed.index for j in observed.columns]
        
        observed_percentages = observed_flat / np.sum(observed_flat) * 100
        expected_percentages = expected_flat / np.sum(expected_flat) * 100
        
        x = np.arange(len(categories))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, observed_percentages, width, label='Observed', color=CONFIG['charts']['data_colors'][0], alpha=0.7, edgecolor=get_darker_color(CONFIG['charts']['data_colors'][0],30),linewidth=1)
        bars2 = ax.bar(x + width/2, expected_percentages, width, label='Expected', color=CONFIG['charts']['data_colors'][1], alpha=0.7, edgecolor=get_darker_color(CONFIG['charts']['data_colors'][1],30),linewidth=1)
        
        ax.set_ylabel('%')
        ax.set_title('Observed vs Expected %')
        ax.set_xticks(x)
        ROTATION = 15
        ax.tick_params(axis='x', rotation=ROTATION)
        ax.legend()
        ax.set_xticklabels(categories)
        
        # Add value labels on bars
        def autolabel(bars):
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height, f'{height:.1f}%',ha='center', va='bottom', rotation=15,color=CONFIG['charts']['font_color'])

        autolabel(bars1)
        autolabel(bars2)

    MAX_CATEGORIES,SIGNIFICANCE = 30,alpha

    try:
        if y in ['None',None,'none']:
            y = [item for item in list(df.select_dtypes(include=['object']).columns) if len(df[item].unique()) < MAX_CATEGORIES][0]
        if category in ['None',None,'none']:
            category = [item for item in list(df.select_dtypes(include=['object']).columns) if len(df[item].unique()) < MAX_CATEGORIES][1] 
    except Exception as e:
        print(f"Error due to require atleast 2 'object' type columns: {e}")
    
    data = df[[category,y]].copy()
    contingency_table = pd.crosstab(data[category],data[y])
    observed = contingency_table.values
    chi2, p, dof, expected = stats.chi2_contingency(observed)

    fig, axs = plt.subplots(4,1,figsize=(5,15),dpi=75)
    plot_chi2_distribution(ax=axs[0],chi2=chi2,dof=dof,alpha=SIGNIFICANCE)
    plot_cell_contributions(ax=axs[1], observed=contingency_table, expected=expected)
    plot_percentage_comparison(ax=axs[2], observed=contingency_table, expected=expected)
    sns.countplot(data=data,x=y,hue=category,ax=axs[3])
    #sns.barplot(data=df,x=category,y=y,ax=axs[1])

    # Get data for plotting
    #education_levels = counts.index
    #male_counts = counts["male"]
    #female_counts = counts["female"]
    #x = np.arange(len(data.index))  # Bar positions

    # Create the barplot
    '''BAR_WIDTH = 0.4
    for i,cat in enumerate(df[x].unique()):
        COLOR_INDEX = i % len(CONFIG['charts']['data_colors'])
        try:
            plt.bar(x - BAR_WIDTH/2, data.loc[data,x], width=BAR_WIDTH, label=cat, color=CONFIG['charts']['data_colors'])
        except Exception as e:
            print(e)    '''

    
    # Add labels, title, and legend
    #plt.xticks(ticks=x, labels=data[])

    return {
        'output':fig,
        'output_type':'chart',
        'title':f"Test if '{category}'(=Category) effecting '{y}'(=Category) via 'Chi^2' Test",
        'table':contingency_table,
        'args':{
            'df':['df'],
            'y':[f"'{item}'" for item in list(df.select_dtypes(include=['object']).columns) if len(df[item].unique()) < MAX_CATEGORIES],
            'category':[f"'{item}'" for item in list(df.select_dtypes(include=['object']).columns) if len(df[item].unique()) < MAX_CATEGORIES],
            'alpha':[0.05,0.01,0.1]
            }
        }       
    

def get_time_plot(df:pd.DataFrame,y:str='None',x:str='None',by:str='None',model:str='arima',autoreg_order:int=0,int_order:int=0,ma_order:int=0):

    if y in [None,'none','None']:
        y = list(df.select_dtypes(include=['number']).columns)[0]  
        
    if x not in [None,'none','None']:
        data = df.set_index(x).copy()   
    else:
        data = df.copy()    

    data = data[[y,by]].dropna() if by not in [None,'none','None'] else data[[y]].dropna()
    data['pred'] = None
      
    POINT_SIZE = 2 if len(data) > 1000 else 4 if len(data) > 200 else 5
    ALPHA = 0.7  
    fig, axs = plt.subplots(2,1,figsize=(6,12),dpi=75)

    if by in [None,'none','None']: # no categories
        axs[0].plot(data.index, data[y], 
                linestyle='-', 
                linewidth=1,
                color=CONFIG['charts']['data_colors'][0],
                alpha=ALPHA
                )

        if model == 'arima':
            model = sm.tsa.ARIMA(data[y], order=(autoreg_order,int_order,ma_order))
        elif model == 'sarima':
            model = sm.tsa.SARIMAX(data[y], order=(1,1,1), seasonal_order=(1,1,1,12))
    
        results = model.fit()
        data['pred'] = results.predict(start=data.index[0], end=data.index[-1])     

        axs[0].plot(data.index, data['pred'], label="pred",color=get_darker_color(CONFIG['charts']['data_colors'][0],40))
    
    else: # categories
        for i,cat in enumerate(data[by].unique()):
            COLOR_INDEX = i % len(CONFIG['charts']['data_colors'])
            axs[0].plot(data.loc[data[by]==cat,[y]],
                linestyle='-', 
                linewidth=1, 
                color=CONFIG['charts']['data_colors'][COLOR_INDEX],
                alpha=ALPHA
                )                
            #axs[0].plot(data.index, data['pred'], label="pred",color=get_darker_color(CONFIG['charts']['data_colors'][COLOR_INDEX],60))

    return {
        'output':fig,
        'output_type':'chart',
        'title':f'TimeSeries of "{y}" by {data.index.name}',
        'table':pd.DataFrame(data.head(10)),
        'args':{
            'df':['df'],
            'y':[f'"{item}"' for item in list(df.select_dtypes(include=['number']).columns)],
            'x':[f'"None"'] + [f'"{item}"' for item in list(df.select_dtypes(include=['number','datetime64[ns]']).columns)],
            'by':[f'"None"'] + [f'"{item}"' for item in list(df.select_dtypes(include=['object']).columns) if len(df[item].unique()) < 16],
            'model':[f"'arima'",f"'sarima'"],
            'autoreg_order':[0,1,2,3,4,5],
            'int_order':[0,1,2],
            'ma_order':[0,1,2,3,4,5]
            }
        }


    #return pd.DataFrame()
    data = {
        'Q1':df[x].quantile(0.25),
        'mean':df[x].mean(),
        'std':df[x].std(),
        '(Q2) median':df[x].median(),
        'Q3':df[x].quantile(0.75),
        'IQR':df[x].quantile(0.75)-df[x].quantile(0.25)
        #'lower_whisker':max(df[x].min(), Q1 - 1.5 * IQR),
        #'upper_whisker':min(df[x].max(), Q3 + 1.5 * IQR)
        }
    return pd.DataFrame(data=data)   
def set_rf_model(df:pd.DataFrame,y:str='None'):

    fig, axs = plt.subplots(1,1,figsize=(4,4),dpi=75)

    return {
        'output':fig,
        'output_type':'chart',
        'title':f'Random Forest "{y}"',
        'table':pd.DataFrame(df.head()),
        'args':{
            'df':['df'],
            'y':[f'"{item}"' for item in df.columns]
            }
        }
  