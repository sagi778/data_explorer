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
    def update_list_content(list_box:ttk.Treeview,parent_dir:str):
        list_box.delete(*list_box.get_children())
        for file in ["..."] + get_dir(parent_dir):
            if get_file_type(file) in CONFIG['supported_files']:
                list_box.insert('','end',text=file,tags=('data_file',))
            else:
                list_box.insert('','end',text=file)

        list_box.tag_configure("data_file", foreground="green")
            
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
            file_label.config(text=file_name,foreground='green')
            DATA_TABLE['path'] = path
            DATA_TABLE['file_name'] = file_name
            DATA_TABLE['df'] = pd.read_csv(DATA_TABLE['path'])
            print(f"Loading {DATA_TABLE['path']}")
            global col_explorer,data_view
            data_view.grid_forget()
            data_view = data_file_view(root,df=DATA_TABLE['df'],CONFIG=CONFIG)
            data_view.grid(row=0,column=2)

            col_explorer.grid_forget()
            col_explorer = column_explorer(root,df=DATA_TABLE['df'])
            col_explorer.grid(row=0,column=1)
            return
        
        return    
    def get_item(event): # double clicked    
        #print(">> pressed double click on files list") # monitor
        entry_string = exp_entry.get()
        current_path = '/'.join(entry_string.split('/')[:-1]) + '/'
        file_name = exp_lb.item(exp_lb.selection(),'text')
        #file_name = exp_lb.get(0,END)[exp_lb.curselection()[0]]

        exp_entry.delete(0,END)
        exp_entry.insert(0,f"{current_path}{file_name}")
        get_content(event)
        #print(f">> type={file_type}, go_back={file_path.endswith('...')}") # monitor

    frame = ttk.Frame(parent)

    file_label = ttk.Label(frame,font=(CONFIG['explorer']['font'],CONFIG['explorer']['cmd_font_size']))
    if CONFIG['file'] == '':
        file_label.config(text='>> Pick data file:',foreground=CONFIG['explorer']['cmd_font_color'])

    file_label.pack(side=TOP)
    
    exp_entry = ttk.Entry(frame,style='Custom.TEntry')
    exp_entry.insert(0,path)
    exp_entry.bind('<Return>',get_content) # Bind the return click event to the entry
    exp_entry.pack(side=TOP,pady=2,fill=X)

    exp_lb = ttk.Treeview(frame,height=45,show='tree',style="Custom.Treeview")
    exp_lb.column("#0", width=300)  # set treeview width
    update_list_content(list_box=exp_lb,parent_dir=exp_entry.get())

    exp_lb.pack(side=TOP,padx=2,fill='both',expand=True)
    exp_lb.bind('<Double-1>',get_item) # Bind the double click event to the listbox

    return frame
def column_explorer(parent,df:pd.DataFrame,CONFIG=CONFIG):
    def update_list_content(list_box:ttk.Treeview,parent_dir:str):
        list_box.delete(*list_box.get_children())
        for column in df.columns.tolist():
            data_type = str(df[column].dtype)
            if data_type in CONFIG['data_types']:
                list_box.insert('','end',text=f"({data_type}) {column}",tags=(data_type,))
            else:
                list_box.insert('','end',text=f"({data_type}) {column}")    

        for dtype in CONFIG['data_types']:                  
            list_box.tag_configure(dtype, foreground=CONFIG['data_types'][dtype]) 
    def get_content(event): # pressed return 
        print(f"pressed Return")        
    def get_column(event): # double clicked    
        #print(f'Loading {exp_lb.get(0,END)[exp_lb.curselection()[0]]}')
        column_string = exp_lb.item(exp_lb.selection(),'text')
        column_name = column_string[column_string.find(')')+2:]
        exp_entry.delete(0,END)
        exp_entry.insert(0,column_name)

        global data_view
        data_view.grid_forget()
        data_view = column_view(root,df=df,column=column_name)
        data_view.grid(row=0,column=2)
    def get_column_menu(event): # right click
        print("right clicked")
        # Create a Menu
        menu = Menu(root, tearoff=0)
        menu.add_command(label="Add x")
        menu.add_command(label="Add y")
        menu.post(event.x_root,event.y_root)

    frame = ttk.Frame(parent)

    label_title = f"{DATA_TABLE['file_name'].split('.')[0]}.columns:" if DATA_TABLE['file_name']!=None else ''
    file_label = ttk.Label(frame,text=label_title,font=(CONFIG['explorer']['font'],CONFIG['explorer']['cmd_font_size']))
    file_label.pack(side=TOP,fill=X)
    
    exp_entry = ttk.Entry(frame,style='Custom.TEntry')
    exp_entry.insert(0,'')
    exp_entry.bind('<Return>',get_content) # Bind the return click event to the entry
    exp_entry.pack(side=TOP,pady=2,fill=X)

    exp_lb = ttk.Treeview(frame,height=45,show='tree',style="Custom.Treeview")
    exp_lb.column("#0", width=200)  # set treeview width
    update_list_content(list_box=exp_lb,parent_dir=exp_entry.get())

    #exp_lb = Listbox(frame,width=35,height=45,font=(CONFIG['font'],CONFIG['font_size']))
    #update_list_content(list_box=exp_lb,df=df)

    exp_lb.bind('<Double-1>',get_column) # Bind the double click event to the listbox
    exp_lb.bind("<Button-3>", get_column_menu)
    exp_lb.pack(side=TOP,fill=X)

    return frame

# data file viewer
def data_table(parent,df:pd.DataFrame,CONFIG=CONFIG): 
    MAX_HEIGHT = min(25*len(df),600)
    MAX_WIDTH = min(1000,8*5*len(df.columns))
    frame = ttk.Frame(parent,width=MAX_WIDTH,height=MAX_HEIGHT)
    frame.pack_propagate(False)  # Prevent frame from resizing to fit its contents

    tree = ttk.Treeview(frame,style="Custom.Treeview")

    tree["columns"] = list(df.columns)
    for col in df.columns:
        COLUMN_WIDTH = np.max([len(col)] + [len(str(item)) for item in df[col]])
        tree.column(col, width=8*COLUMN_WIDTH)
        tree.heading(col, text=col)

    tree.column("#0", width=len(df.index)*6)  # Adjust index column width


    for i, row in df.iterrows():
        tree.insert("", "end", text=i, values=row.tolist())
           
    tree_scroll = ttk.Scrollbar(frame, orient="horizontal", command=tree.xview)
    tree.configure(xscrollcommand=tree_scroll.set)

    tree_scroll.pack(side="bottom", fill="x")
    tree.pack(fill="both", expand=True)

    return frame
def data_file_view(parent,df:pd.DataFrame,sample=5,CONFIG=CONFIG):

    if DATA_TABLE['file_name'] != None:
        frame = Label(parent,text=DATA_TABLE['file_name'],bg=CONFIG['background'],padx=1,pady=1)
        
        shape_entry = ttk.Entry(frame,style='Custom.TEntry')
        shape_entry.insert(0,f"{DATA_TABLE['file_name'].split('.')[0]}.shape")
        shape_entry.pack(side=TOP,fill=X)
        shape_label = ttk.Label(frame,text=f"{df.shape}")
        shape_label.pack(side=TOP,fill=X,pady=10)

        prev_entry = ttk.Entry(frame,style='Custom.TEntry')
        prev_cmd = f"{DATA_TABLE['file_name'].split('.')[0]}.show({sample})" if len(df) > 100 else f"{DATA_TABLE['file_name'].split('.')[0]}.show()"
        prev_entry.insert(0,prev_cmd)
        prev_entry.pack(side=TOP,fill=X)

        if len(df) < 101:
            prev_df = df
        else:
            prev_df = pd.concat([df.head(sample),
                                pd.DataFrame({column:['. . .'] for column in df.columns.to_list()},index=['. . .']),
                                df.tail(sample)])
            
        prev = data_table(frame,df=prev_df)
        prev.pack(side=TOP,fill=X,pady=10)
        
        desc_entry = ttk.Entry(frame,style='Custom.TEntry')
        desc_entry.insert(0,f"{DATA_TABLE['file_name'].split('.')[0]}.describe().T")
        desc_entry.pack(side=TOP,fill=X)
        desc_table = data_table(frame,df=df.describe().T)
        desc_table.pack(side=TOP,fill=X)
    else:
        frame = ttk.Frame(parent)

    return frame

# column viewer
def column_desc(parent,df:pd.DataFrame,column=None,CONFIG=CONFIG):
    frame = ttk.Label(parent)

    disp_string = f"column[{column}].describe()" if len(column) < 12 else f"column[{column[0:5]}...{column[-4:]}].describe()"
    label = ttk.Label(frame,text=disp_string,font=(CONFIG['font'],CONFIG['font_size']))
    label.grid(row=0,column=0)

    lb = Listbox(frame,width=35,height=14,font=(CONFIG['font'],CONFIG['font_size']-1))
    
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

    lb.grid(row=2,column=0)

    return frame
def column_preview(parent,df:pd.DataFrame,column=None,sample_size=5,CONFIG=CONFIG):
    frame = ttk.Label(parent)

    prev_string = f"column[{column}].preview()" if len(column) < 12 else f"column[{column[0:5]}...{column[-4:]}].preview()"
    label = ttk.Label(frame,text=prev_string,font=(CONFIG['font'],CONFIG['font_size']),width=30)
    label.grid(row=0,column=0)

    lb_frame = Frame(frame,bg=CONFIG['border_color'],padx=1,pady=1)
    lb = Listbox(lb_frame,width=30,height=sample_size*2+1,font=(CONFIG['font'],CONFIG['font_size']))
    
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
        frame = ttk.Label(parent)  

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
    
    frame = ttk.Label(parent)
    
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


def column_view(parent,df:pd.DataFrame,column=None,CONFIG=CONFIG):

    # clean prior charts from memory
    try:
        plt.clf()
        plt.close()
    except:
        pass

    frame = ttk.Label(parent,text=column)
    #print(f"column: {column}") # monitor
    desc = column_desc(frame,df=df,column=column)
    desc.grid(row=0,column=0)

    dist = column_dist(frame,df=df,column=column)
    dist.grid(row=0,column=1)

    prev = column_preview(frame,df=df,column=column,sample_size=5)
    prev.grid(row=1,column=0)

    return frame



# operation func
def close_window():
    if messagebox.askyesno("Quit", "Do you want to quit?"):
        print('bye bye...')
        root.destroy()

# main
root = ThemedTk(theme='clearlooks') # equilux/yaru/clearlooks/breeze/arc/adapta/classic/default/scidgreen/scidblue/aqua/breeze-dark/awdark
#root.configure(bg='white')
root.geometry(f"{int(root.winfo_screenwidth()/1.1)}x{int(root.winfo_screenheight()/1.1)}")
root.title('Data Explorer')

style = ttk.Style()
style.configure("Custom.Treeview",
                font=(CONFIG['explorer']['font'],CONFIG['explorer']['font_size']),
                foreground=CONFIG['explorer']['font_color'],  # Text color
                background=CONFIG['explorer']['background'],  # Background color
                #fieldbackground=CONFIG['explorer']['selection_color'],  # Field background color (for cells)
                #bordercolor=CONFIG['explorer']['frame_color'],
                borderwidth=7,
                rowheight=20)  # Row height
style.configure('Custom.TEntry', 
                foreground=CONFIG['explorer']['font_color'],        # Text color
                background=CONFIG['explorer']['background'],  # Background color
                borderwidth=2,           # Border width
                relief='sunken',         # Border style
                font=(CONFIG['explorer']['font'],CONFIG['explorer']['cmd_font_size']),      # Font
                padding=(1,1,1,1))    # Padding (top, right, bottom, left)


file_explorer = file_explorer(root,path=CONFIG['main_path'])
file_explorer.grid(row=0,column=0)

col_explorer = column_explorer(root,df=DATA_TABLE['df'])
col_explorer.grid(row=0,column=1)

data_view = data_file_view(root,df=None)
data_view.grid(row=0,column=2,padx=25)

#table_exp = data_table(root,df=DATA_TABLE['df'],sample=10)
#table_exp.grid(row=0,column=2)  

root.protocol("WM_DELETE_WINDOW",close_window)
root.mainloop()