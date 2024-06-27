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
def load_config(file_path):
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

# charts
def get_scatter_plot(parent,df:pd.DataFrame,x:str=None,y:str=None):

    # Create a CTkFrame to hold the chart
    frame = ctk.CTkFrame(parent)

    data = df[[x,y]].copy().dropna()
    # Create a Seaborn plot
    fig, ax = plt.subplots()
    sns.scatterplot(data=df,x=x,y=y)
    
    # Create a canvas to embed the plot
    canvas = FigureCanvasTkAgg(fig, master=frame)
    canvas.draw()
    canvas.get_tk_widget().pack(fill="both", expand=True)
    
    return frame

# data frame func
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