from tkinter import *
from tkinter import ttk
import customtkinter as ctk 
from PIL import Image, ImageTk
from ttkthemes import ThemedTk
from tkinter import messagebox

import os
import pandas as pd
pd.set_option('display.max_columns', None) 
pd.set_option('display.expand_frame_repr', False)
import numpy as np
from scipy.stats import linregress,gaussian_kde,shapiro
from sklearn.neighbors import LocalOutlierFactor
import json
import os
import re
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns

import traceback



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

# load config files
CURRENT_PATH = '\\'.join(os.path.abspath(__file__).split('\\')[:-1]) + '\\'
CONFIG = load_json(f'{CURRENT_PATH}config.json')

# charts
plt.rcParams['axes.facecolor'] = CONFIG['charts']['background']  # background
plt.rcParams['axes.edgecolor'] = CONFIG['charts']['frame_color']  # frame & axes


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

    data = pd.DataFrame(data).sort_values(by=['Non-Nulls']).reset_index(drop=True)  

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
    data["pearson's_skewness"] = 3*(data['mean'] - data['50%'])/data['std']

    if show != 'all':
        data = data[data.index == show]

    return {
        'output':data,
        'output_type':'table',
        'args':{'df':['df'],'show':["'all'"] + [f'"{column}"' for column in numeric_columns]}
        }
def get_categorical_desc(df:pd.DataFrame,show='all'):
    categorical_columns = [col for col in df.columns if str(df[col].dtype) in ['object','category','bool']]
    df = df[categorical_columns]
    data = {'column':[],'type':[],'unique':[],'mode':[],'mode_occurances':[],'mode%':[]}#,'Non-Nulls%':[]}
    for column in df.columns:
        data['column'].append(column)
        data['type'].append(str(df[column].dtype))
        data['unique'].append(len(df[column].unique().tolist()))
        data['mode'].append(df[column].mode()[0])
        data['mode_occurances'].append(len(df[df[column]==df[column].mode()[0]]))
        data['mode%'].append(f"{100*len(df[df[column]==df[column].mode()[0]])/len(df):.2f}%")

    data = pd.DataFrame(data).reset_index(drop=True)

    if show != 'all':
        data = data[data['column'] == show]

    return {
        'output':data,
        'output_type':'table',
        'args':{'df':['df'],'show':["'all'"] + [f'"{column}"' for column in categorical_columns]}
        }

def get_correlations(df:pd.DataFrame,in_chart:bool=False):
    data = df[df.select_dtypes(include=['number']).columns].corr()

    st = pd.DataFrame({'Summary Table':['no data']})

    if in_chart:
        fig, ax = plt.subplots(figsize=(10,10),dpi=80)
        colors = [CONFIG['charts']['data_colors'][1],CONFIG['charts']['data_colors'][0],CONFIG['charts']['data_colors'][1]] # green,purple,green
        cmap = LinearSegmentedColormap.from_list('custom_diverging', colors, N=100)

        im = ax.imshow(data, cmap=cmap, interpolation='nearest', vmin=-1, vmax=1)
        fig.colorbar(im, ax=ax, fraction=0.05)
        ax.set_xticks(range(len(data.columns)))
        ax.set_yticks(range(len(data.columns)))
        ax.set_xticklabels(data.columns, rotation=45, ha='right')
        ax.set_yticklabels(data.columns)

        for i in range(len(data.columns)):
            for j in range(len(data.columns)):
                text = ax.text(j, i, f'{data.iloc[i, j]:.2f}',
                            ha='center', va='center', color='black')

        plt.tight_layout() 

        # summary table
        #st = pd.DataFrame(data.unstack())   
        #print(st)               

    return {
        'output':data if in_chart==False else fig,
        'output_type':'chart' if in_chart==True else 'table',
        'title':'Correlations Heatmap',
        'table':st,
        'args':{'df':['df'],'in_chart':[True,False]}
        }
def get_relation_plot(df:pd.DataFrame,x:str=None,y:str=None,by:str=None,reg_type='linear',exclude_outliers="None"):

    if x==None:
        x = list(df.select_dtypes(include=['number']).columns)[0] 
    if y==None:
        y = list(df.select_dtypes(include=['number']).columns)[1]     

    data = df[[x,y]].dropna().copy() if by in [None,"None",'none'] else df[[x,y,by]].dropna().copy()

    POINT_SIZE = 4 if len(data) > 1000 else 8 if len(data) > 200 else 20
    ALPHA = 0.5 if len(data) > 1000 else 0.8 if len(data) > 200 else 1

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
            'Category':['all'],
            'Relation_type':[reg_type],
            'R^2':[],
            'Std_err':[],
            'Pred_Func':[]
        }

        COLOR_INDEX = CONFIG['charts']['data_colors'][0]
        HIST_BINS = min(50,data.shape[0])

        # Plot histograms
        axs[0, 0].hist(data[x], alpha=ALPHA, bins=HIST_BINS, edgecolor=CONFIG['charts']['frame_color'], color=COLOR_INDEX)
        axs[1, 1].hist(data[y], alpha=ALPHA, bins=HIST_BINS,orientation='horizontal', edgecolor=CONFIG['charts']['frame_color'], color=COLOR_INDEX)

        if exclude_outliers not in ['None',None,'none']: # outliers - need to fix
            PERCENTAGE = float(exclude_outliers[:exclude_outliers.find('%')])/100
            lof = LocalOutlierFactor(n_neighbors=min(int(0.1*len(data)),20), contamination=PERCENTAGE)
            data['outlier'] = lof.fit_predict(data)
            norm_data = data[data.outlier==1]
            outliers_data = data[data.outlier==-1]

            axs[1,0].plot(norm_data[x],norm_data[y],
                linestyle='none', 
                linewidth=1, 
                alpha=ALPHA,
                marker='o', 
                markersize=POINT_SIZE, 
                markerfacecolor= COLOR_INDEX, 
                markeredgecolor=CONFIG['charts']['frame_color'] 
            )

            axs[1,0].plot(outliers_data[x],outliers_data[y],
                linestyle='none', 
                linewidth=1, 
                alpha=0.9,
                marker='o', 
                markersize=POINT_SIZE, 
                markerfacecolor= COLOR_INDEX, 
                markeredgecolor='red'
            )

        else:
            axs[1,0].plot(data[x],data[y],
                linestyle='none', 
                linewidth=1, 
                alpha=ALPHA,
                marker='o', 
                markersize=POINT_SIZE, 
                markerfacecolor= COLOR_INDEX, 
                markeredgecolor=CONFIG['charts']['frame_color'] 
            )

        if reg_type == 'linear':
            slope, intercept, r_value, p_value, std_err = linregress(data[x], data[y])
            regression_line = slope * data[x] + intercept
            axs[1,0].plot(data[x], regression_line, color=get_darker_color(COLOR_INDEX,30), label=f'Linear Fit: $y={slope:.2f}x+{intercept:.2f}$')
            summary_table['R^2'].append(f"{r_value**2:.4f}")
            summary_table['Std_err'].append(std_err)
            summary_table['Pred_Func'].append(f"y={slope:.4f}x+{intercept:.4f}")
    
    else: # chart by categories
        #print(f"colors:{len(CONFIG['charts']['data_colors'])}, categories: {len(data[by].unique())}") # monitor
        summary_table = {
            'Category':[],
            #'Relation_type':[reg_type]*len(data[by].unique()),
            'R^2':[],
            'Std_err':[],
            'Pred_Func':[]
            }

        for i,category in enumerate(data[by].unique()):

            summary_table['Category'].append(category)
            COLOR_INDEX = i if i <= len(data[by].unique()) else i-len(data[by].unique())
            HIST_BINS = min(50,data.loc[data[by]==category,x].shape[0])

            axs[0, 0].hist(data.loc[data[by]==category,x], bins=HIST_BINS, alpha=ALPHA,edgecolor=CONFIG['charts']['frame_color'], color=CONFIG['charts']['data_colors'][COLOR_INDEX])
            axs[1, 1].hist(data.loc[data[by]==category,y], bins=HIST_BINS, alpha=ALPHA,orientation='horizontal', edgecolor=CONFIG['charts']['frame_color'], color=CONFIG['charts']['data_colors'][COLOR_INDEX])
            
            axs[1,0].plot(data.loc[data[by]==category,x],data.loc[data[by]==category,y],
                linestyle='none', 
                linewidth=1, 
                alpha=ALPHA,
                marker='o', 
                markersize=POINT_SIZE, 
                markerfacecolor=CONFIG['charts']['data_colors'][COLOR_INDEX], 
                markeredgecolor=CONFIG['charts']['frame_color'] 
            )

            if reg_type == 'linear':
                slope, intercept, r_value, p_value, std_err = linregress(data.loc[data[by]==category,x],data.loc[data[by]==category,y])
                regression_line = slope * data.loc[data[by]==category,x] + intercept
                axs[1,0].plot(data.loc[data[by]==category,x], regression_line, color=CONFIG['charts']['data_colors'][COLOR_INDEX], label=f'Linear Fit: $y={slope:.2f}x+{intercept:.2f}$')
                summary_table['R^2'].append(f"{r_value**2:.4f}")
                summary_table['Std_err'].append(std_err)
                summary_table['Pred_Func'].append(f"y={slope:.4f}x+{intercept:.4f}")
            else:
                summary_table['R^2'].append(None)
                summary_table['Std_err'].append(None)
                summary_table['Pred_Func'].append(None)    

    try: # monitor
        sum_table = pd.DataFrame(summary_table) 
        print(summary_table)
    except Exception as e:
        print(e)
        sum_table = pd.DataFrame({'Error occured creating summary table':e},index=[0])
            
    plt.tight_layout() 
    return {
        'output':fig,
        'output_type':'chart',
        'title':f'Behavior of "{y}"(=Numeric) by Variance of "{x}"(=Numeric)',
        'table':sum_table,
        'args':{'df':['df'],
                'x':[f'"{item}"' for item in list(df.select_dtypes(include=['number']).columns)],
                'y':[f'"{item}"' for item in list(df.select_dtypes(include=['number']).columns)],
                'by':["None"] + [f'"{item}"' for item in list(df.select_dtypes(include=['object']).columns) if len(df[item].unique()) < 16],
                'reg_type':["None",f'"linear"'],
                'exclude_outliers':['"None"','"0.3%"','"0.5"','"1%"','"5%"','"10%"']}
        }
def get_dist_plot(df:pd.DataFrame,x=str,outliers="none"):
    
    fig, axs = plt.subplots(2,1,figsize=(6,3),dpi=75,sharex=True)
    
    if x==None:
        x = list(df.select_dtypes(include=['number']).columns)[0]  

    data = df[[x]].dropna().copy()
    POINT_SIZE = 5 if len(data) > 1000 else 8 if len(data) > 200 else 20
    ALPHA = 0.1 if len(data) > 1000 else 0.4 if len(data) > 200 else 0.6
    
    MEAN = data[x].mean()
    MEDIAN = data[x].median()
    STD = data[x].std()
    Q1 = data[x].quantile(0.25)
    Q3 = data[x].quantile(0.75)
    LCL = data[x].quantile(0.003)
    UCL = data[x].quantile(0.997)
    IQR = Q3 - Q1
    lower_whisker = max(data[x].min(), Q1 - 1.5 * IQR)
    upper_whisker = min(data[x].max(), Q3 + 1.5 * IQR)

    sns.boxplot(data[x],orient="h", color="white", linewidth=1, showfliers=False, ax=axs[0])
    if outliers == "IQR":
        outliers_data = data.loc[(data[x] < Q1 - 1.5 * IQR)|(data[x] > Q3 + 1.5 * IQR),x]
        no_outliers_data = data.loc[(data[x] >= lower_whisker) & (data[x] <= upper_whisker),x]
    elif '%' in outliers:     
        PERCENTAGE = float(outliers[:outliers.find('%')])
        lower_threshold = data[x].quantile((PERCENTAGE/2)/100)
        upper_threshold = data[x].quantile(1-((PERCENTAGE/2)/100))
        outliers_data = data.loc[(data[x] < lower_threshold)|(data[x] > upper_threshold),x]
        no_outliers_data = data.loc[(data[x] > lower_threshold)&(data[x] < upper_threshold),x]   
    elif outliers == "None":    
        no_outliers_data = data[x]
        outliers_data = []

    sns.stripplot(no_outliers_data,ax=axs[0],orient='h',alpha=ALPHA,size=POINT_SIZE,linewidth=0.5,color=CONFIG['charts']['data_colors'][0],edgecolor=CONFIG['charts']['frame_color'],jitter=0.35)   
    if outliers != "None":
        sns.stripplot(outliers_data,ax=axs[0],orient='h',alpha=0.4,size=POINT_SIZE,linewidth=0.5,color='red',edgecolor=CONFIG['charts']['frame_color'],jitter=0.3)    

    axs[1].hist(data[x],bins=min(len(data),50),color=CONFIG['charts']['data_colors'][0],edgecolor=CONFIG['charts']['frame_color'], alpha=0.7)

    axs[1].axvline(MEDIAN, color='red', linestyle='-', label=f'Median = {MEDIAN:.2f}', linewidth=1)
    axs[1].text(MEDIAN,0,"median", horizontalalignment="center", verticalalignment="top", transform=axs[1].get_xaxis_transform(), rotation=45,color='red')
    axs[1].axvline(MEAN, color='green', linestyle='-', label=f'Mean = {MEAN:.2f}', linewidth=1)
    axs[1].text(MEAN, 0, "mean", horizontalalignment="center", verticalalignment="top", transform=axs[1].get_xaxis_transform(), rotation=45,color='green')
    axs[1].axvline(LCL, color='purple', linestyle='--', label=f'0.3% Threshold = {LCL:.2f}', linewidth=1)
    axs[1].text(LCL, 0, "-3*std", horizontalalignment="center", verticalalignment="top", transform=axs[1].get_xaxis_transform(), rotation=45,color='purple')
    axs[1].axvline(UCL, color='purple', linestyle='--', label=f'99.7% Threshold = {UCL:.2f}', linewidth=1)
    axs[1].text(UCL, 0, f"3*std", horizontalalignment="center", verticalalignment="top", transform=axs[1].get_xaxis_transform(), rotation=45,color='purple')
    
        
    if len(data[x].unique()) > 100: # adding density plot
        kde_x = np.linspace(data[x].values.min(), data[x].values.max(),100)
        kde_y = gaussian_kde(data[x].values)(np.linspace(data[x].values.min(), data[x].values.max(),100))
        axs[1].twinx().plot(kde_x,kde_y, color=get_darker_color(CONFIG['charts']['data_colors'][0],30), label='Density', linewidth=1)
        #axs[1].twinx().set_yticks(np.linspace(0,1,4))

    for side in ['top','bottom','right','left']: 
        axs[0].spines[side].set_linewidth(1)
        axs[1].spines[side].set_linewidth(1)
    
    plt.tight_layout()
    axs[1].set_ylabel("Count")
    axs[1].twinx().set_ylabel("Density")
    axs[1].legend()


    # summary table
    st = pd.DataFrame({
        'count':[len(data[x])],
        'outliers':[len(outliers_data)],
        'min':[f"{data[x].min():.4f}"],
        'mean':[f"{MEAN:.4f}"],
        'median':[f"{MEDIAN:.4f}"],
        'std':[f"{STD:.4f}"],
        'max':[f"{data[x].max():.4f}"],
        'interquartile_range':[f"{Q1:.4f} - {Q3:.4f}"],
        'skewness':[f"{3*(MEAN-MEDIAN)/STD:.4f}"]
        })

    return {
        'output':fig,
        'output_type':'chart',
        'title':f'"{x}" Values distribution:',
        'table':st,
        'args':{
            'df':['df'],
            'x':[f'"{item}"' for item in list(df.select_dtypes(include=['number']).columns)],
            'outliers':[f'"None"',f'"IQR"',f'"0.3%"',f'"0.5%"',f'"1%"',f'"5%"',f'"10%"']
            }
        }
def get_compare(df:pd.DataFrame,y='None',category='None',show_outliers='None'):
    
    if category in ['None',None]:
        category = list(df.select_dtypes(include=['object']).columns)[0] 
    if y in ['None',None]:
        y = list(df.select_dtypes(include=['number']).columns)[0]  

    data = df[[category,y]].dropna().copy()
    WIDTH,HEIGHT = 15,5
    fig, ax = plt.subplots(figsize=(WIDTH,HEIGHT),dpi=75)

    st = {'category':[],'count':[],'min':[],'mean':[],'median':[],'std':[],'max':[],'outliers':[]}
    sns.boxplot(data=data,x=category,y=y,color="white", linewidth=1, showfliers=False)

    for i,cat in enumerate(data[category].unique()):
        
        dp_y = data.loc[data[category]==cat,y]
        dp_x = data.loc[data[category]==cat,category]

        st['category'].append(cat)
        st['count'].append(len(dp_y))
        st['min'].append(min(dp_y))
        st['mean'].append(np.mean(dp_y))
        st['median'].append(np.median(dp_y))
        st['std'].append(np.std(dp_y))
        st['max'].append(max(dp_y))

        POINT_SIZE = 6 if len(dp_y) < 400 and len(dp_x) < 10 else 4
        ALPHA = 0.3 if len(dp_y) > 1000 else 0.4 if len(dp_y) > 200 else 0.5
        DP_COLOR = CONFIG['charts']['data_colors'][i % len(CONFIG['charts']['data_colors'])] 

        if '%' in show_outliers:
            PERCENTAGE = float(show_outliers[:show_outliers.find('%')])
            LCL = dp_y.quantile((PERCENTAGE/2)/100)
            UCL = dp_y.quantile(1 - ((PERCENTAGE/2)/100))
            dp_ouliers = dp_y[(dp_y > UCL)|(dp_y < LCL)] 
            dp_y = dp_y[(dp_y <= UCL) & (dp_y >= LCL)] 
            st['outliers'].append(len(dp_ouliers))
            sns.stripplot(y=dp_ouliers,x=len(dp_ouliers)*[cat],alpha=0.9,size=POINT_SIZE,linewidth=0.7,color=DP_COLOR,edgecolor='red',jitter=0.35) 
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
        'title':f"Compare '{y}'(=Numeric) column by '{category}' (=Category):\n",
        'table':pd.DataFrame(st),
        'args':{
            'df':['df'],
            'y':[f'"{item}"' for item in list(df.select_dtypes(include=['number']).columns)],
            'category':[f'"{item}"' for item in list(df.select_dtypes(include=['object']).columns)],
            'show_outliers':[f"'None",f"'IQR'",f"'0.3%'",f"'0.5%'",f"'1%'",f"'5%'",f"'10%'"]
            }
        }

# column func
def get_quantiles(df:pd.DataFrame,x:str):
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