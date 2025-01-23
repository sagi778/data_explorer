from tkinter import *
from tkinter import ttk
import customtkinter as ctk 
from PIL import Image, ImageTk
from ttkthemes import ThemedTk
from tkinter import messagebox

import os
import pandasql as psql
import pandas as pd
pd.set_option('display.max_columns', None) 
pd.set_option('display.max_rows', None)
pd.set_option('display.expand_frame_repr', False)
import numpy as np
from scipy import stats
from scipy.stats import linregress,gaussian_kde,shapiro,ttest_ind
from sklearn.neighbors import LocalOutlierFactor
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
def get_shape(df:pd.DataFrame):
    return {
        'output':pd.DataFrame(data=df.shape,columns=['#'],index=['rows','columns']).T,
        'output_type':'table',
        'args':{'df':['df']}
        }  
def get_columns(df:pd.DataFrame):
    return {
        'output':pd.DataFrame(df.columns,columns=['Column']),
        'output_type':'table',
        'args':{'df':['df']}
        }  
def get_columns_info(df:pd.DataFrame,show='all'): 
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

    return {
        'output':data,
        'output_type':'table',
        'args':{'df':['df'],'show':["'all'"] + [f"'{col}'" for col in list(df.columns)]}
        }  
def get_numerics_desc(df:pd.DataFrame,show='all'):
    data = df.describe().T
    numeric_columns = list(data.index)
    data['count'] = data['count'].astype(int)
    data["skewness"] = (data['mean'] - data['50%'])/data['std']

    if show != 'all':
        data = data[data.index == show]

    return {
        'output':data,
        'output_type':'table',
        'args':{'df':['df'],'show':["'all'"] + [f'"{column}"' for column in numeric_columns]}
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
            data['prob_outliers'].append(None)
            data['outlier_items'].append(None)
            data['outliers_occurance_probability'].append(None)
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
def get_preview(df:pd.DataFrame,rows=5,end='head'):
    if rows >= len(df):
        data = df
    elif end == 'head':
        data = df.head(rows)
    elif end == 'tail':
        data = df.tail(rows)
    else:
        data = pd.DataFrame()            
    return {
        'output':data,
        'output_type':'table',
        'args':{'df':['df'],'rows':[3,5,10,25],'end':[f"'head'",f"'tail'"]}
        }
def get_data(df:pd.DataFrame,sql_string:str):
    try:
        output_type = 'table'
        data = psql.sqldf(sql_string)
    except Exception as e:
        output_type = 'text' 
        data = f"\n>>> Error processing SQL:\n{e}\n"
           
    #print(f" >>> output:{data}\n >>> output_type:{output_type}") # monitor
    return {
        'output':data,
        'output_type':output_type,
        'args':{'df':['df'],'sql_string':[f"'SELECT * FROM df LIMIT 10'"]}
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
def get_relation_plot(df:pd.DataFrame,x:str='None',y:str='None',by:str='None',reg_type='linear',exclude_outliers="None"):

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
            'std_err':[],
            'excluded_outliers':[]
        }

        COLOR_INDEX = CONFIG['charts']['data_colors'][0]
        HIST_BINS = min(50,data.shape[0])

        # Plot histograms
        axs[0, 0].hist(data[x], alpha=ALPHA-0.2, bins=HIST_BINS, edgecolor=CONFIG['charts']['frame_color'], color=COLOR_INDEX)
        axs[1, 1].hist(data[y], alpha=ALPHA-0.2, bins=HIST_BINS,orientation='horizontal', edgecolor=CONFIG['charts']['frame_color'], color=COLOR_INDEX)

        if exclude_outliers not in ['None',None,'none']:
            PERCENTAGE = float(exclude_outliers[:exclude_outliers.find('%')])/100
            lof = LocalOutlierFactor(contamination=PERCENTAGE)
            data['inlier'] = lof.fit_predict(data)
            summary_table['excluded_outliers'].append(len(data[data.inlier==-1]))

            axs[1,0].plot(data.loc[data.inlier==-1,x],data.loc[data.inlier==-1,y], # outliers dp
                linestyle='none', 
                linewidth=1.5, 
                alpha=0.9,
                marker='o', 
                markersize=POINT_SIZE, 
                markerfacecolor= CONFIG['charts']['data_colors'][0],
                markeredgecolor='red'
            )
        else:
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
            axs[1,0].plot(data.loc[data.inlier==1,x], regression_line, color=get_darker_color(COLOR_INDEX,30), label=f'Linear Fit: $y={slope:.2f}x+{intercept:.2f}$')
            summary_table['r^2'].append(f"{r_value**2:.4f}")
            summary_table['std_err'].append(std_err)
            summary_table['pred_func'].append(f"y={slope:.4f}x+{intercept:.4f}")
        else:
            summary_table['r^2'].append(None)
            summary_table['std_err'].append(None)
            summary_table['pred_func'].append(None)    
    
    else: # chart by categories
        summary_table = {
            'category':[],
            'count':[],
            'type':[],
            'pred_func':[],
            'r^2':[],
            'std_err':[],
            'excluded_outliers':[]
            }

        for i,category in enumerate(data[by].unique()):
            
            summary_table['type'].append(reg_type)
            summary_table['category'].append(category)
            summary_table['count'].append(len(data[data[by]==category]))
            COLOR_INDEX =  i % len(CONFIG['charts']['data_colors'])
            HIST_BINS = min(50,data.loc[data[by]==category,x].shape[0])

            axs[0, 0].hist(data.loc[data[by]==category,x], bins=HIST_BINS, alpha=ALPHA-0.2,edgecolor=CONFIG['charts']['frame_color'], color=CONFIG['charts']['data_colors'][COLOR_INDEX])
            axs[1, 1].hist(data.loc[data[by]==category,y], bins=HIST_BINS, alpha=ALPHA-0.2,orientation='horizontal', edgecolor=CONFIG['charts']['frame_color'], color=CONFIG['charts']['data_colors'][COLOR_INDEX])
            
            if exclude_outliers not in ['None',None,'none']:
                PERCENTAGE = float(exclude_outliers[:exclude_outliers.find('%')])/100
                lof = LocalOutlierFactor(contamination=PERCENTAGE)
                data['inlier'] = lof.fit_predict(data[[x,y]])
                summary_table['excluded_outliers'].append(len(data[(data.inlier==-1)&(data[by]==category)]))

                axs[1,0].plot(data.loc[(data[by]==category)&(data.inlier==-1),x],data.loc[(data[by]==category)&(data.inlier==-1),y], # outliers dp
                linestyle='none', 
                linewidth=1.5, 
                alpha=0.9,
                marker='o', 
                markersize=POINT_SIZE, 
                markerfacecolor=CONFIG['charts']['data_colors'][COLOR_INDEX],
                markeredgecolor='red'
            )
            else:
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
                summary_table['std_err'].append(std_err)
                summary_table['pred_func'].append(f"y={slope:.4f}x+{intercept:.4f}")
            else:
                summary_table['r^2'].append(None)
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
                'exclude_outliers':['"None"','"0.3%"','"0.5%"','"1%"','"5%"']}
        }
def get_dist_plot(df:pd.DataFrame,x:str=None,by:str=None,outliers="none",exclude_outliers=False):
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
            'IQR':f"[{data.quantile(0.25)}:{data.quantile(0.75)}]",
            'lower_whisker':max(data.min(), data.quantile(0.25) - 1.5 * (data.quantile(0.75) - data.quantile(0.25))),
            'upper_whisker':min(data.max(), data.quantile(0.75) + 1.5 * (data.quantile(0.75) - data.quantile(0.25)))
        }
    def set_vlines(ax,stats:{},keys:{}):
        for key,color in keys.items():
            #print(f"{key}:{stats[key]}(color={color})")
            LABEL,VALUE,COLOR = key,stats[key],color
            ax.axvline(VALUE, color=COLOR, linestyle='-',linewidth=2) #label=f'{LABEL} = {VALUE:.2f}'
            ax.text(VALUE, 0, LABEL, horizontalalignment="center", verticalalignment="top", transform=ax.get_xaxis_transform(), rotation=45,color=COLOR)
    def set_data(df:pd.DataFrame,x:str,by:str=None,outliers=None,exclude_outliers:bool=False):
        data = pd.DataFrame(columns=['no_data'])
        match exclude_outliers:
            case False|'False'|'false':
                data = df[[x]].dropna().copy() if by in [None,'none','None'] else df[[x,by]].dropna().copy()  
            case True|'True'|'true':
                match by:
                    case None|'none'|'None':
                        STATS = get_stats(df[x])
                        match outliers:
                            case  None|'none'|'None':
                                data = df[[x]].dropna().copy()
                            case 'IQR':
                                data = df.loc[(df[x] < STATS['upper_whisker'])&(df[x] > STATS['lower_whisker']),[x]]  
                            case _:
                                PERCENTAGE = float(outliers[:outliers.find('%')])   
                                lower_threshold = df[x].quantile((PERCENTAGE/2)/100)
                                upper_threshold = df[x].quantile(1-((PERCENTAGE/2)/100))
                                data = df.loc[(df[x] > lower_threshold)&(df[x] < upper_threshold),[x]]
                    case _:
                        match outliers:
                            case None|'none'|'None':
                                data = df[[x]].dropna().copy()
                            case 'IQR':
                                data = pd.DataFrame()
                                for item in df[by].unique():
                                    STATS = get_stats(df.loc[df[by].to_numpy()==item,x])
                                    data = pd.concat([data,df.loc[(df[by].to_numpy()==item) & (df[x].to_numpy() < STATS['upper_whisker'])&(df[x].to_numpy() > STATS['lower_whisker']),[x,by]]])    
                            case _:
                                data = pd.DataFrame()
                                PERCENTAGE = float(outliers[:outliers.find('%')]) 
                                for item in df[by].unique():
                                    lower_threshold = df.loc[df[by].to_numpy()==item,x].quantile((PERCENTAGE/2)/100)
                                    upper_threshold = df.loc[df[by].to_numpy()==item,x].quantile(1-((PERCENTAGE/2)/100))
                                    data = pd.concat([data,df.loc[(df[by]==item) & (df[x].to_numpy() > lower_threshold) & (df[x].to_numpy() < upper_threshold),[x]]])
        
        return data 

    if x in [None,'none','None']:
        x = list(df.select_dtypes(include=['number']).columns)[0]  

    data = set_data(df=df,x=x,by=by,outliers=outliers,exclude_outliers=exclude_outliers)
    POINT_SIZE = 5 if len(data) > 1000 else 8 if len(data) > 200 else 9
    ALPHA = 0.1 if len(data) > 1000 else 0.4 if len(data) > 200 else 0.6
    HEIGHT = 3 if by in [None,'nona','None'] else int(1.5*len(data[by].unique()))
    fig, axs = plt.subplots(2,1,figsize=(6,HEIGHT),dpi=75,sharex=True,gridspec_kw={'height_ratios': [HEIGHT,3]})
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
        'skewness':[f"{STATS['mean']-STATS['median']:.2f}"]
        }

    if by in [None,'none','None']: # no categories
        axs[1].hist(data[x],bins=min(len(data),50),color=CONFIG['charts']['data_colors'][0],edgecolor=get_darker_color(CONFIG['charts']['data_colors'][0],70), alpha=0.3)
        STATS = get_stats(data[x])
        set_vlines(ax=axs[1],stats=STATS,keys={'mean':'green','median':'red','-3*std':'purple','+3*std':'purple'})
        sns.boxplot(data[x],orient="h",color="white",linewidth=1, showfliers=False, ax=axs[0])

        # add outliers
        if outliers == "IQR":
            outliers_data = data.loc[(data[x] < STATS['q1'] - 1.5 * STATS['iqr'])|(data[x] > STATS['q3'] + 1.5 * STATS['iqr']),x]
            no_outliers_data = data.loc[(data[x] >= STATS['lower_whisker']) & (data[x] <= STATS['upper_whisker']),x]
        elif '%' in outliers:     
            PERCENTAGE = float(outliers[:outliers.find('%')])
            lower_threshold = data[x].quantile((PERCENTAGE/2)/100)
            upper_threshold = data[x].quantile(1-((PERCENTAGE/2)/100))
            outliers_data = data.loc[(data[x] < lower_threshold)|(data[x] > upper_threshold),x]
            no_outliers_data = data.loc[(data[x] > lower_threshold)&(data[x] < upper_threshold),x]   
        elif outliers == "None":    
            no_outliers_data = data[x]
            outliers_data = []

        sns.stripplot(no_outliers_data,ax=axs[0],orient='h',alpha=ALPHA,size=POINT_SIZE,linewidth=0.5,color=CONFIG['charts']['data_colors'][0],edgecolor=get_darker_color(CONFIG['charts']['data_colors'][0],70),jitter=0.35)   
        if outliers not in [None,'none',"None"]:
            sns.stripplot(outliers_data,ax=axs[0],orient='h',alpha=0.4,size=POINT_SIZE,linewidth=0.5,color='red',edgecolor=get_darker_color(CONFIG['charts']['frame_color'],70),jitter=0.3) 

        if len(data[x].unique()) > 100: # adding density plot
            kde_x = np.linspace(data[x].values.min(), data[x].values.max(),100)
            kde_y = gaussian_kde(data[x].values)(np.linspace(data[x].values.min(), data[x].values.max(),100))
            axs[1].twinx().plot(kde_x,kde_y, color=get_darker_color(CONFIG['charts']['data_colors'][0],30), label='Density', linewidth=1)

    else: # by categories
        sns.boxplot(x=data[x], y=data[by],color="white",linewidth=1, showfliers=False, ax=axs[0])
        for i,cat in enumerate(data[by].unique()):
            STATS = get_stats(data.loc[data[by]==cat,x])
            for stat in st.keys(): # update summary table
                st[stat].append(cat if stat=='category' else STATS[stat])

            COLOR_INDEX = i % len(CONFIG['charts']['data_colors'])
            axs[1].hist(data.loc[data[by]==cat,x],bins=min(len(data),50),color=CONFIG['charts']['data_colors'][COLOR_INDEX],edgecolor=get_darker_color(CONFIG['charts']['data_colors'][COLOR_INDEX],70), alpha=0.3)
            set_vlines(ax=axs[1],stats=STATS,keys={'mean':get_darker_color(CONFIG['charts']['data_colors'][COLOR_INDEX],60),'median':get_darker_color(CONFIG['charts']['data_colors'][COLOR_INDEX],30)})

            # add outliers
            if outliers == "IQR":
                outliers_data = data.loc[(data[by]==cat)&((data[x] < STATS['q1'] - 1.5 * STATS['iqr'])|(data[x] > STATS['q3'] + 1.5 * STATS['iqr'])),[x,by]]
                no_outliers_data = data.loc[(data[by]==cat)&((data[x] >= STATS['lower_whisker']) & (data[x] <= STATS['upper_whisker'])),[x,by]]
            elif '%' in outliers:     
                PERCENTAGE = float(outliers[:outliers.find('%')])
                lower_threshold = data[x].quantile((PERCENTAGE/2)/100)
                upper_threshold = data[x].quantile(1-((PERCENTAGE/2)/100))
                outliers_data = data.loc[(data[by]==cat)&((data[x] < lower_threshold)|(data[x] > upper_threshold)),[x,by]]
                no_outliers_data = data.loc[(data[by]==cat)&((data[x] > lower_threshold)&(data[x] < upper_threshold)),[x,by]]   
            elif outliers == "None":    
                no_outliers_data = data.loc[data[by]==cat,[x,by]]
                outliers_data = []

            ALPHA = 0.1 if len(data[data[by]==cat]) > 1000 else 0.4 if len(data[data[by]==cat]) > 200 else 0.6
            sns.stripplot(x=no_outliers_data[x],y=no_outliers_data[by],ax=axs[0],orient='h',alpha=ALPHA,size=POINT_SIZE,linewidth=0.5,color=CONFIG['charts']['data_colors'][COLOR_INDEX],edgecolor=get_darker_color(CONFIG['charts']['data_colors'][COLOR_INDEX],70),jitter=0.35)   
            if outliers not in [None,'none',"None"]:
                sns.stripplot(x=outliers_data[x],y=outliers_data[by],ax=axs[0],orient='h',alpha=0.4,size=POINT_SIZE,linewidth=0.5,color='red',edgecolor=get_darker_color(CONFIG['charts']['data_colors'][COLOR_INDEX],70),jitter=0.3) 

            if len(data.loc[data[by]==cat,x]) > 100: # adding density plot
                kde_x = np.linspace(data.loc[data[by]==cat,x].values.min(), data.loc[data[by]==cat,x].values.max(),100)
                kde_y = gaussian_kde(data.loc[data[by]==cat,x].values)(np.linspace(data.loc[data[by]==cat,x].values.min(), data.loc[data[by]==cat,x].values.max(),100))
                axs[1].twinx().plot(kde_x,kde_y, color=get_darker_color(CONFIG['charts']['data_colors'][COLOR_INDEX],30), label='Density', linewidth=1)
    
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
            'outliers':[f'"None"',f'"IQR"',f'"0.3%"',f'"0.5%"',f'"1%"',f'"5%"',f'"10%"'],
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
    sns.boxplot(data=data,x=category,y=y,color="white", linewidth=1, showfliers=False,ax = ax)
    
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

        sns.lineplot(x=st['category'],y=st['mean'],color='blue',linewidth=1)    
    
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
  