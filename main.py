from func import *

# load config file
CURRENT_PATH = '\\'.join(os.path.abspath(__file__).split('\\')[:-1]) + '\\'
CONFIG = load_config(f'{CURRENT_PATH}config.json')

# load data
DATA_TABLE = {'path':CONFIG['file'],
              'file_name':None,
              'df':pd.DataFrame() if CONFIG['file']=='' else pd.read_csv(CONFIG['file'])
             }

# widgets
def file_explorer(parent,path:str,CONFIG=CONFIG):
    def update_list_content(list_box:Listbox,parent_dir:str):
            list_box.delete(0,END)
            for file in ["..."] + get_dir(parent_dir):
                    list_box.insert(END,file)
                    if get_file_type(file) in CONFIG['supported_files']:
                        list_box.itemconfig(END,{'fg':'green'}) 
    def get_content(event): # pressed return 
        
        #print("pressed Return on path entry") # monitor
        path = exp_entry.get()
        file_name = path.split('/')[-1]
        file_type = get_file_type(path)

        if file_name == '...':
            parent_dir_path = '/'.join(path.split('/')[:-2]) + '/'
            exp_entry.delete(0,END)
            exp_entry.insert(0,parent_dir_path)
            file_label.config(text='')
            update_list_content(list_box=exp_lb,parent_dir=parent_dir_path)

        if file_type == 'dir':
            update_list_content(list_box=exp_lb,parent_dir=path)
            return
        
        if file_type in CONFIG['supported_files']: 
            file_label.config(text=file_name,fg='green')
            DATA_TABLE['path'] = path
            DATA_TABLE['file_name'] = file_name
            DATA_TABLE['df'] = pd.read_csv(DATA_TABLE['path'])
            print(f"Loading {DATA_TABLE['path']}")
            global col_explorer
            col_explorer.grid_forget()
            col_explorer = column_explorer(root,df=DATA_TABLE['df'])
            col_explorer.grid(row=0,column=1)
            return
        
        return    
    def get_item(event): # double clicked    
        #print(">> pressed double click on files list") # monitor
        entry_string = exp_entry.get()
        current_path = '/'.join(entry_string.split('/')[:-1]) + '/'
        file_name = exp_lb.get(0,END)[exp_lb.curselection()[0]]

        exp_entry.delete(0,END)
        exp_entry.insert(0,f"{current_path}{file_name}")
        get_content(event)
        #print(f">> type={file_type}, go_back={file_path.endswith('...')}") # monitor

    frame = Label(parent,bg=CONFIG['background'],pady=1)

    file_label = Label(frame,bg=CONFIG['background'],font=(CONFIG['font'],CONFIG['font_size']+2),width=45,padx=1,pady=0)
    if CONFIG['file'] == '':
        file_label.config(text='>> Pick data file:',fg='red')

    file_label.pack(side=TOP,pady=2)
    
    entry_frame = Frame(frame,bg=CONFIG['border_color'],pady=1)
    exp_entry = Entry(entry_frame,width=45,background=CONFIG['entry_color'],fg=CONFIG['font_color'],bd=CONFIG['border'],highlightcolor=CONFIG['highlight_color'],highlightthickness=CONFIG['highlight_thick'],font=(CONFIG['font'],CONFIG['font_size']))
    exp_entry.insert(0,path)
    exp_entry.bind('<Return>',get_content) # Bind the return click event to the entry
    exp_entry.pack(padx=1)
    entry_frame.pack(side=TOP,pady=2)

    lb_frame = Frame(frame,bg=CONFIG['border_color'],pady=1)
    exp_lb = Listbox(lb_frame,width=45,height=45,activestyle='none',selectbackground=CONFIG['selection_color'],selectforeground='black',fg=CONFIG['font_color'],background=CONFIG['entry_color'],bd=CONFIG['border'],highlightcolor=CONFIG['highlight_color'],highlightthickness=CONFIG['highlight_thick'],font=(CONFIG['font'],CONFIG['font_size']))
    update_list_content(list_box=exp_lb,parent_dir=exp_entry.get())

    exp_lb.pack(padx=1)
    exp_lb.bind('<Double-1>',get_item) # Bind the double click event to the listbox
    lb_frame.pack(side=TOP,pady=1)

    return frame
def column_explorer(parent,df:pd.DataFrame,CONFIG=CONFIG):
    def update_list_content(list_box:Listbox,df:pd.DataFrame):
        list_box.delete(0,END)
        for column in df.columns.tolist():
            data_type = str(df[column].dtype)
            list_box.insert(END,f"({data_type}) {column}")
            #print(f"data_type={data_type} ? {CONFIG['data_types_colors']}") # monitor
            if data_type in CONFIG['data_types_colors'].keys():
                list_box.itemconfig(END,{'fg':CONFIG['data_types_colors'][data_type]})
         
    def get_content(event): # pressed return 
        print(f"pressed Return")        
    def get_column(event): # double clicked    
        #print(f'Loading {exp_lb.get(0,END)[exp_lb.curselection()[0]]}')
        column_string = exp_lb.get(0,END)[exp_lb.curselection()[0]]
        column_name = column_string[column_string.find(')')+2:]
        exp_entry.delete(0,END)
        exp_entry.insert(0,column_name)

        global col_view
        col_view.grid_forget()
        col_view = column_view(root,df=df,column=column_name)
        col_view.grid(row=0,column=2)
    def get_column_menu(event): # right click
        print("right clicked")
        # Create a Menu
        menu = Menu(root, tearoff=0)
        menu.add_command(label="Add x")
        menu.add_command(label="Add y")
        menu.post(event.x_root,event.y_root)

    frame = Label(parent,bg=CONFIG['background'])

    label_title = f"{DATA_TABLE['file_name'].split('.')[0]}.columns:" if DATA_TABLE['file_name']!=None else ''
    file_label = Label(frame,text=label_title,bg=CONFIG['background'],font=(CONFIG['font'],CONFIG['font_size']+2),width=45)
    file_label.pack(side=TOP)
    
    entry_frame = Frame(frame,bg=CONFIG['border_color'])
    exp_entry = Entry(entry_frame,width=45,background=CONFIG['entry_color'],fg=CONFIG['font_color'],bd=CONFIG['border'],highlightcolor=CONFIG['highlight_color'],highlightthickness=CONFIG['highlight_thick'],font=(CONFIG['font'],CONFIG['font_size']))
    exp_entry.insert(0,'')
    exp_entry.bind('<Return>',get_content) # Bind the return click event to the entry
    exp_entry.pack(padx=1,pady=1)
    entry_frame.pack(side=TOP,pady=2)

    lb_frame = Frame(frame,bg=CONFIG['border_color'])
    exp_lb = Listbox(lb_frame,width=45,height=45,activestyle='none',selectbackground=CONFIG['selection_color'],selectforeground='black',fg=CONFIG['font_color'],background=CONFIG['entry_color'],bd=CONFIG['border'],highlightcolor=CONFIG['highlight_color'],highlightthickness=CONFIG['highlight_thick'],font=(CONFIG['font'],CONFIG['font_size']))
    update_list_content(list_box=exp_lb,df=df)

    exp_lb.pack(padx=1,pady=1)
    exp_lb.bind('<Double-1>',get_column) # Bind the double click event to the listbox
    exp_lb.bind("<Button-3>", get_column_menu)
    lb_frame.pack(side=TOP)

    return frame

# column viewer
def column_desc(parent,df:pd.DataFrame,column=None,CONFIG=CONFIG):
    frame = Label(parent,bg=CONFIG['background'],padx=2,pady=2)

    disp_string = f"column[{column}].describe()" if len(column) < 12 else f"column[{column[0:5]}...{column[-4:]}].describe()"
    label = Label(frame,text=disp_string,bg=CONFIG['background'],font=(CONFIG['font'],CONFIG['font_size']),width=30,padx=1,pady=0)
    label.grid(row=0,column=0)

    lb_frame = Frame(frame,bg=CONFIG['border_color'],padx=1,pady=1)
    lb = Listbox(lb_frame,width=30,height=14,activestyle='none',selectbackground=CONFIG['selection_color'],selectforeground='black',fg=CONFIG['font_color'],background=CONFIG['entry_color'],bd=CONFIG['border'],highlightcolor=CONFIG['highlight_color'],highlightthickness=CONFIG['highlight_thick'],font=(CONFIG['font'],CONFIG['font_size']))
    
    try:
        lb.insert(END,f"rows = {len(df)}")
        lb.insert(END,f"values = {len(df[~df[column].isna()])}")
        lb.insert(END,f"nulls = {len(df[df[column].isna()])}")
        lb.insert(END,'')
        #print(f"dtype: {df[column].dtype}") # monitor
        if df[column].dtype in ['int64','float64']:
            q1, q3 = np.percentile(df[column], [25, 75])

            lb.insert(END,f"mean = {np.mean(df[column])}")
            lb.itemconfig(END,{'fg':CONFIG['charts']['mean_color']})
            lb.insert(END,f"std = {np.std(df[column])}")
            lb.insert(END,f"min = {np.min(df[column])}")
            lb.insert(END,f"5% = {np.percentile(df[column],5)}")
            lb.insert(END,f"25% = {np.percentile(df[column],25)}")
            lb.insert(END,f"50%/median = {np.median(df[column])}")
            lb.itemconfig(END,{'fg':CONFIG['charts']['median_color']})
            lb.insert(END,f"75% = {np.percentile(df[column],75)}")
            lb.insert(END,f"95% = {np.percentile(df[column],95)}")
            lb.insert(END,f"max = {np.max(df[column])}")
            lb.insert(END,f"outliers = {len(df.loc[(df[column] > q3+1.5*(q3-q1))|(df[column] < q1-1.5*(q3-q1)),column])}")
            lb.itemconfig(END,{'fg':'red'})
        else:
            column_dict = df[column].value_counts().to_dict()
            lb.insert(END,f"uniques = {len(np.unique(df[column]))}")    
            for item in column_dict:
                lb.insert(END,f"{item} = {column_dict[item]}")
    except:
        pass        

    lb.pack()
    lb_frame.grid(row=2,column=0,padx=5,pady=5)

    return frame
def column_preview(parent,df:pd.DataFrame,column=None,sample_size=5,CONFIG=CONFIG):
    frame = Label(parent,bg=CONFIG['background'],padx=2,pady=2)

    prev_string = f"column[{column}].preview()" if len(column) < 12 else f"column[{column[0:5]}...{column[-4:]}].preview()"
    label = Label(frame,text=prev_string,bg=CONFIG['background'],font=(CONFIG['font'],CONFIG['font_size']),width=30,padx=1,pady=0)
    label.grid(row=0,column=0)

    lb_frame = Frame(frame,bg=CONFIG['border_color'],padx=1,pady=1)
    lb = Listbox(lb_frame,width=30,height=sample_size*2+1,activestyle='none',selectbackground=CONFIG['selection_color'],selectforeground='black',fg=CONFIG['font_color'],background=CONFIG['entry_color'],bd=CONFIG['border'],highlightcolor=CONFIG['highlight_color'],highlightthickness=CONFIG['highlight_thick'],font=(CONFIG['font'],CONFIG['font_size']))
    
    try:
        for i in df.index[0:sample_size]:
            lb.insert(END,f"{i}: {df.loc[i,column]}")

        lb.insert(END,"...")

        for i in df.index[-1*sample_size:]:
            lb.insert(END,f"{i}: {df.loc[i,column]}")    
    except:
        pass        

    lb.pack()
    lb_frame.grid(row=2,column=0,padx=5,pady=5)

    return frame
def column_dist(parent,df:pd.DataFrame,column=None,CONFIG=CONFIG):
    def histplot(parent,df:pd.DataFrame,column=None,CONFIG=CONFIG): 
        # Create a frame to contain the chart
        frame = Label(parent,bg=CONFIG['background'],padx=0,pady=1)  

        f, ax = plt.subplots(figsize=(6,2))

        # customizations
        plt.gca().set_facecolor(CONFIG['charts']['background'])
        ax.spines['top'].set_color(CONFIG['charts']['frame_color']) 
        ax.spines['right'].set_color(CONFIG['charts']['frame_color'])  
        ax.spines['bottom'].set_color(CONFIG['charts']['frame_color'])  
        ax.spines['left'].set_color(CONFIG['charts']['frame_color'])
        ax.set_xlabel(ax.get_xlabel(), fontdict={'fontsize': CONFIG['charts']['font_size'],'color':CONFIG['charts']['font_color']})
        ax.set_ylabel(ax.get_ylabel(), fontdict={'fontsize': CONFIG['charts']['font_size'],'color':CONFIG['charts']['font_color']})
        ax.tick_params(axis='x', colors=CONFIG['charts']['font_color'])
        ax.tick_params(axis='y', colors=CONFIG['charts']['font_color'])

        sns_plot = sns.histplot(data=df[column],color=CONFIG['charts']['data_colors'][0],edgecolor="0.5",linewidth=.5)
        mean,median,std = np.mean(df[column]),np.median(df[column]),np.std(df[column])
        plt.axvline(mean, color=CONFIG['charts']['mean_color'], linestyle='solid', linewidth=1)
        plt.axvline(median, color=CONFIG['charts']['median_color'], linestyle='solid', linewidth=1)
        plt.axvline(mean+3*std, color='red', linestyle='--', linewidth=1)
        plt.axvline(mean-3*std, color='red', linestyle='--', linewidth=1)

        canvas = FigureCanvasTkAgg(sns_plot.get_figure(),master=frame)
        canvas.draw()
        canvas.get_tk_widget().pack() 
            
        return frame
    def boxplot(parent,df:pd.DataFrame,column=None,CONFIG=CONFIG): 
        # Create a frame to contain the chart
        frame = Label(parent,bg=CONFIG['background'],padx=0,pady=1)  

        f, ax = plt.subplots(figsize=(6,1))

        # customizations
        plt.gca().set_facecolor(CONFIG['charts']['background'])
        ax.spines['top'].set_color(CONFIG['charts']['frame_color']) 
        ax.spines['right'].set_color(CONFIG['charts']['frame_color'])  
        ax.spines['bottom'].set_color(CONFIG['charts']['frame_color'])  
        ax.spines['left'].set_color(CONFIG['charts']['frame_color'])
        ax.set_xlabel(ax.get_xlabel(), fontdict={'fontsize': CONFIG['charts']['font_size'],'color':CONFIG['charts']['font_color']})
        ax.set_ylabel(ax.get_ylabel(), fontdict={'fontsize': CONFIG['charts']['font_size'],'color':CONFIG['charts']['font_color']})
        ax.tick_params(axis='x', colors=CONFIG['charts']['font_color'])
        ax.tick_params(axis='y', colors=CONFIG['charts']['font_color'])

        q1, q3 = np.percentile(df[column], [25, 75])
        iqr = q3 - q1
        upper_whisker = q3 + 1.5 * iqr
        lower_whisker = q1 - 1.5 * iqr

        outliers_df = df.loc[(~df[column].isna())&((df[column] < lower_whisker)|(df[column] > upper_whisker)),column]
        inliers_df = df.loc[(~df[column].isna()&((df[column] < upper_whisker)&(df[column] > lower_whisker))),column]
        MARKER_SIZE = 3 if len(outliers_df) > 50 else 5
        ALPHA = 0.4 if len(outliers_df) > 50 else 0.6
        sns_plot = sns.stripplot(x=outliers_df,color='red',size=MARKER_SIZE,marker='o',jitter=0.3,linewidth=0.5,alpha=ALPHA,ax=ax)
        sns_plot = sns.stripplot(x=inliers_df,color=CONFIG['charts']['data_colors'][0],size=MARKER_SIZE,marker='o',jitter=0.3,linewidth=0.5,alpha=ALPHA-0.2,ax=ax)

        sns_plot = sns.boxplot(data=df[column], orient='h',fliersize=0,linewidth=0.5,dodge=True,whis=1.5,ax=ax)

        canvas = FigureCanvasTkAgg(sns_plot.get_figure(),master=frame)
        canvas.draw()
        canvas.get_tk_widget().pack() 
            
        return frame
    
    frame = Label(parent,bg=CONFIG['background'],padx=1,pady=1)
    
    try:
        data_type = df[column].dtype
        column_exist = column in df.columns
    except:
        data_type = None
        column_exist = False 

    if data_type in ['int64','float64'] and column_exist:
        histplot(frame,df=df,column=column).pack()
        boxplot(frame,df=df,column=column).pack()  

    return frame

def column_scatter(parent,df:pd.DataFrame,column=None,CONFIG=CONFIG): 
    # Create a frame to contain the chart
    frame = Label(parent,bg=CONFIG['background'],padx=2,pady=2)

    try:
        data_type = df[column].dtype
        column_exist = column in df.columns
    except:
        data_type = None
        column_exist = False    

    if data_type in ['int64','float64'] and column_exist:
        f, ax = plt.subplots(figsize=(7,6))

        # customizations
        plt.gca().set_facecolor(CONFIG['charts']['background'])
        ax.spines['top'].set_color(CONFIG['charts']['frame_color']) 
        ax.spines['right'].set_color(CONFIG['charts']['frame_color'])  
        ax.spines['bottom'].set_color(CONFIG['charts']['frame_color'])  
        ax.spines['left'].set_color(CONFIG['charts']['frame_color'])
        ax.set_xlabel(ax.get_xlabel(), fontdict={'fontsize': CONFIG['charts']['font_size'],'color':CONFIG['charts']['font_color']})
        ax.set_ylabel(ax.get_ylabel(), fontdict={'fontsize': CONFIG['charts']['font_size'],'color':CONFIG['charts']['font_color']})
        ax.tick_params(axis='x', colors=CONFIG['charts']['font_color'])
        ax.tick_params(axis='y', colors=CONFIG['charts']['font_color'])

        sns_plot = sns.scatterplot(y=df[column],x=df.index,s=5,alpha=0.6,edgecolor=CONFIG['charts']['font_color'])

        canvas = FigureCanvasTkAgg(sns_plot.get_figure(),master=frame)
        canvas.draw()
        canvas.get_tk_widget().pack() 
        
    return frame

def column_view(parent,df:pd.DataFrame,column=None,CONFIG=CONFIG):

    # clean prior charts from memory
    try:
        plt.clf()
        plt.close()
    except:
        pass

    frame = Label(parent,text=column,bg=CONFIG['background'],padx=1,pady=1)
    #print(f"column: {column}") # monitor
    desc = column_desc(frame,df=df,column=column)
    desc.grid(row=0,column=0)

    dist = column_dist(frame,df=df,column=column)
    dist.grid(row=0,column=1)

    prev = column_preview(frame,df=df,column=column,sample_size=5)
    prev.grid(row=1,column=0)

    return frame


def data_table(parent,df:pd.DataFrame,sample=10,CONFIG=CONFIG):
    frame = Frame(parent,background=CONFIG['entry_color'])

    col,row = 0,0
    for column in [df.index] + df.columns.tolist():
        if type(column) != pd.RangeIndex:
            e = Entry(frame,width=20,background=CONFIG['table']['background'],fg=CONFIG['font_color'],bd=CONFIG['border'],highlightcolor=CONFIG['highlight_color'],highlightthickness=1,font=(CONFIG['table']['font'],CONFIG['table']['font_size']))
            e.insert(0,column)
            e.grid(row=0,column=col)
        for index in range(len(df.index[0:sample])):
            e = Entry(frame,width=20,background=CONFIG['table']['background'],fg=CONFIG['font_color'],bd=CONFIG['border'],highlightcolor=CONFIG['highlight_color'],highlightthickness=1,font=(CONFIG['table']['font'],CONFIG['table']['font_size']))
            e.insert(index, index) if col == 0 else e.insert(index,df.loc[row,column])
            e.grid(row=row+1,column=col)
            row += 1

        row = 0 
        col += 1

    return frame

# operation func
def close_window():
    if messagebox.askyesno("Quit", "Do you want to quit?"):
        print('bye bye...')
        root.destroy()


# main
root = Tk()
root.configure(bg='white')
root.geometry(f"{int(root.winfo_screenwidth()/1.1)}x{int(root.winfo_screenheight()/1.1)}")
root.title('Explorer')

file_explorer = file_explorer(root,path=CONFIG['main_path'])
file_explorer.grid(row=0,column=0)

col_explorer = column_explorer(root,df=DATA_TABLE['df'])
col_explorer.grid(row=0,column=1)

col_view = column_view(root,df=None,column='')
col_view.grid(row=0,column=2)

#table_exp = data_table(root,df=DATA_TABLE['df'],sample=10)
#table_exp.grid(row=0,column=2)  

root.protocol("WM_DELETE_WINDOW",close_window)
root.mainloop()