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
    
# data frame func
def get_columns_info(df:pd.DataFrame): 
    #data = pd.DataFrame(columns=['column','type','Non-Nulls','Non-Nulls%'])
    data = {'column':[],'type':[],'unique':[],'Non-Nulls':[],'Non-Nulls%':[]}
    for column in df.columns:
        data['column'].append(column)
        data['type'].append(str(df[column].dtype))
        data['unique'].append(len(df[column].unique().tolist()))
        data['Non-Nulls'].append(len(df[column].isna()))
        data['Non-Nulls%'].append(f"{round(len(df[column].isna())*100/len(df),2)}%")
                         
    return pd.DataFrame(data)    