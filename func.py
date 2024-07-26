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
import json
import os
import re
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib import pyplot as plt
import seaborn as sns



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
# visual settings
sns.set_theme(palette="pastel")
sns.set_style(
    style='darkgrid',
    rc = {
        'axes.facecolor':CONFIG['charts']['background'],
        'axes.edgecolor': get_darker_color(CONFIG['charts']['background'],percentage=8),
        'axes.gridcolor': get_darker_color(CONFIG['charts']['background'],percentage=50),
        "axes.linewidth": 1,
        "grid.linewidth": 1,
    }
        )

def get_scatter_plot(parent,df:pd.DataFrame,x:str=None,y:str=None):

    # Create a CTkFrame to hold the chart
    frame = ctk.CTkFrame(parent)

    data = df[[x,y]].dropna().copy()
    # Create a Seaborn plot
    fig, ax = plt.subplots()
    sns.scatterplot(data=df,x=x,y=y)
    
    # Create a canvas to embed the plot
    canvas = FigureCanvasTkAgg(fig, master=frame)
    canvas.draw()
    canvas.get_tk_widget().pack(fill="both", expand=True)
    
    return frame
def get_dist_plot(df:pd.DataFrame,x:str):
    #fig = plt.figure(figsize=(6,5),dpi=90)
    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=(7,4),dpi=75)
    data = df[[x,by]] if by!=None else df[[x]]
    POINT_SIZE = 3 if len(data) > 1000 else 8 if len(data) > 200 else 20
    ALPHA = 0.05 if len(data) > 1000 else 0.4 if len(data) > 200 else 0.6
    
    for col in data.columns:
        if col == by:
            continue

        MEAN = data[col].mean()
        MEDIAN = data[col].median()
        STD = data[col].std()
        Q1 = data[col].quantile(0.25)
        Q3 = data[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_whisker = max(data[col].min(), Q1 - 1.5 * IQR)
        upper_whisker = min(data[col].max(), Q3 + 1.5 * IQR)
        outliers = data[((data[col] < Q1 - 1.5 * IQR) & (data[col] > MEAN - 3*STD)) | ((data[col] > Q3 + 1.5 * IQR) & (data[col] < MEAN + 3*STD))]

        sns.boxplot(
            data=data,
            ax=ax1,
            #hue=by,
            width=0.4,
            linewidth=1,
            #orient='h',
            dodge=True,
            saturation=0.5,
            fliersize=0
            )
        sns.stripplot(
            data=data[(data[col] <= upper_whisker) & (data[col] >= lower_whisker)],
            ax=ax1,
            #hue=by,
            orient='h',
            size=POINT_SIZE,
            jitter=0.3,
            edgecolor='#212F3D',
            linewidth=0.5,
            alpha=ALPHA
            )
        sns.stripplot( # outliers
            data=outliers,
            ax=ax1,
            #hue=by,
            orient='h',
            size=POINT_SIZE,
            jitter=0.2,
            edgecolor='#212F3D',
            linewidth=0.5,
            alpha=ALPHA + 0.1
            )
        sns.stripplot( # 3sigma outliers
            data=data[(data[col] > MEAN + 3*STD) | (data[col] < MEAN - 3*STD)],
            ax=ax1,
            #hue=by,
            orient='h',
            size=POINT_SIZE,
            jitter=0.1,
            edgecolor='red',
            linewidth=0.5,
            alpha=0.8
            )    
        
        sns.stripplot(
            x=[MEDIAN],
            y=[col],
            ax=ax1,
            #hue=by,
            color='#FA8072',
            edgecolor='#212F3D',
            legend=False,
            size=min(POINT_SIZE*3,10),
            linewidth=1,
            marker='D'
            )
        sns.stripplot(
            x=[MEAN],
            y=[col],
            hue=by,
            ax=ax1,
            color='#58D68D',
            edgecolor='#212F3D',
            legend=False,
            size=min(POINT_SIZE*3,10),
            linewidth=1,
            marker='o'
            )
        sns.histplot(
            data=data, 
            hue=by,
            ax=ax2,
            #kde=True, 
            edgecolor=".5",
            linewidth= .5,
            log_scale=False,
            stat="count"
            )
        ax3 = ax2.twinx()
        sns.histplot(
            data=data, 
            hue=by,
            ax=ax3,
            kde=True, 
            edgecolor=".5",
            linewidth= .5,
            log_scale=False,
            stat="probability"
            )
        
    plt.tight_layout()
    return fig

# data frame func
def get_columns(df:pd.DataFrame):
    return pd.DataFrame(df.columns,columns=['Column'])
def get_columns_info(df:pd.DataFrame): 
    #data = pd.DataFrame(columns=['column','type','Non-Nulls','Non-Nulls%'])
    data = {'column':[],'type':[],'unique':[],'Non-Nulls':[],'Non-Nulls%':[]}
    for column in df.columns:
        data['column'].append(column)
        data['type'].append(str(df[column].dtype))
        data['unique'].append(len(df[column].unique().tolist()))
        data['Non-Nulls'].append(len(df[~df[column].isna()]))
        data['Non-Nulls%'].append(f"{round(len(df[~df[column].isna()])*100/len(df),2)}%")
                         
    return pd.DataFrame(data).sort_values(by=['Non-Nulls']).reset_index(drop=True)    

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