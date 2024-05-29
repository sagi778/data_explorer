from func import *

# load config file
CURRENT_PATH = '\\'.join(os.path.abspath(__file__).split('\\')[:-1]) + '\\'
CONFIG = load_config(f'{CURRENT_PATH}config.json')

# load data
DATA_TABLE = {'path':CONFIG['file'],
              'file_name':None,
              'df':pd.DataFrame() if CONFIG['file']=='' else pd.read_csv(CONFIG['file'])
             }

# fundemental widgets
def new_tab(notebook, text: str = '', icon_path: str = None):
    """Create a new tab in the notebook."""
    frame = ctk.CTkFrame(notebook,fg_color='transparent')

    if icon_path is not None:
        image = ctk.PhotoImage(file=icon_path)
        notebook.add(frame, text=text, image=image, compound=ctk.LEFT)
    else:
        notebook.add(frame, text=text)

    return frame    
def save_story(cmd:ttk.Entry): 
    
    global story_tab
    ent = ttk.Entry(story_tab)
    cmd_string = '.'.join(cmd.get().split('.')[1:])

    if 'shape' in cmd_string:
        dim_data(story_tab,df=DATA_TABLE['df']).pack(side=TOP)
    elif 'show' in cmd_string:
        preview_data(story_tab,df=DATA_TABLE['df'],tab='story').pack(side=TOP)
    elif 'describe' in cmd_string:
        describe_data(story_tab,df=DATA_TABLE['df']).pack(side=TOP)
def new_command(parent,cmd_string:str='',data_object:str=['df','column']):

    def get_output(data_object:str,cmd:str):
        output_string = 'Unknown Command.'

        if data_object == 'df' or data_object == DATA_TABLE['file_name'][:DATA_TABLE['file_name'].find('.')]:
            df = DATA_TABLE['df']
            if cmd == 'shape':
                output_string = f"{df.shape}"
            elif cmd == 'info()':    
                output_string = f"{get_columns_info(df)}"    
            elif cmd == 'describe().T':
                output_string = f"{df.describe().T}"    
            elif cmd == 'columns':
                output_string = f"{df.columns.tolist()}"
            elif "head(" in cmd and cmd[-1] == ')': 
                AMOUNT = int(cmd[cmd.find('(')+1:cmd.find(')')])
                output_string = f"{df.head(AMOUNT)}"  
            elif "tail(" in cmd and cmd[-1] == ')': 
                AMOUNT = int(cmd[cmd.find('(')+1:cmd.find(')')])
                output_string = f"{df.tail(AMOUNT)}"         

        return output_string        
    def get_textbox_height(output_string:str,new_output_string:str=None) -> int:
        if new_output_string != '...':
            return output_string.count("\n")*25
        else:
            return 2  
    def show_result(output_string:str):
        btn_text = 'v' if cap.cget('text') == '>>' else '>>'
        cap.configure(text=btn_text)

        new_output_string = '...' if cap.cget('text') == 'v' else output_string 
        res.configure(state='normal')
        res.delete('1.0','end-1c')
        res.insert('1.0',new_output_string)
        res.configure(state='disabled',height=get_textbox_height(output_string,new_output_string))   
    def set_command(event):
        cmd_string = ent.get()
        data_object = cmd_string.split('.')[0] 
        cmd = ''.join(cmd_string.split('.')[1:])
        param = int(cmd[cmd.find('(')+1:cmd.find(')')]) if ('(' in cmd and ')' in cmd) else None
        print(f"Set command: data_object={data_object}, cmd={cmd}, param={param}") # monitor
        
        print(data_object)
        output_string = get_output(data_object=data_object,cmd=cmd)
        res.configure(state='normal') # adjust text area height
        res.delete('1.0',END)
        res.insert('1.0',output_string)
        res.configure(state='disabled',height=get_textbox_height(output_string))   


    cmd = '.'.join(cmd_string.split('.')[1:]) # idenify command type
    output_string = get_output(data_object,cmd)  # get output string

    frame = ctk.CTkFrame(parent,corner_radius=5,fg_color='white',width=1300,border_width=1,border_color=CONFIG['code_block']['frame_color'])

    code = ctk.CTkFrame(frame,fg_color='white')
    ent = ctk.CTkEntry(code,border_width=1,corner_radius=5,fg_color=CONFIG['code_block']['background'],font=(CONFIG['code_block']['font'],CONFIG['code_block']['font_size']+2))
    ent.insert(0,cmd_string)
    ent.bind('<Return>',set_command) # Bind the return click event to the entry
    ent.pack(side=LEFT,padx=2,pady=2,fill=X,expand=True)
    btn = ctk.CTkButton(code,width=40,border_color=CONFIG['button']['frame_color'],border_width=1,hover_color=CONFIG['button']['hover_color'],fg_color=CONFIG['button']['color'],text_color=CONFIG['button']['font_color'],text='+ <Story>',command=lambda: save_story(cmd_string))
    btn.pack(side=LEFT,padx=1,pady=2)
    code.pack(side=TOP,fill=X,padx=2,pady=2)

    result_frame = ctk.CTkFrame(frame,corner_radius=5,fg_color='white')
    cap = ctk.CTkButton(result_frame,width=35,text='>>',fg_color='white',hover_color='#F1F1F1',text_color='green',command=lambda: show_result(output_string))
    cap.pack(side=LEFT,padx=1,pady=1)

    res = ctk.CTkTextbox(result_frame,wrap='none',fg_color='white',border_color='white',text_color='black',font=(CONFIG['code_block']['font'],CONFIG['code_block']['font_size']+1),bg_color='transparent')
    res.pack(side=LEFT,fill=X,padx=2,pady=2,expand=True)

    res.insert("1.0", output_string)
    res.configure(height=get_textbox_height(output_string),state='disabled') # adjust text area height
    result_frame.pack(side=TOP,fill=X,padx=2,pady=2,expand=True)

    return frame

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
            file_label.configure(text='')
            update_list_content(list_box=exp_lb,parent_dir=parent_dir_path)

        if file_type == 'dir':
            update_list_content(list_box=exp_lb,parent_dir=path)
            return
        
        if file_type in CONFIG['supported_files']: 
            file_label.configure(text=file_name,text_color='green')
            DATA_TABLE['path'] = path
            DATA_TABLE['file_name'] = file_name
            DATA_TABLE['df'] = pd.read_csv(DATA_TABLE['path'])
            print(f"Loading {DATA_TABLE['path']}")
            global col_explorer,data_view

            try:
                col_explorer.pack_forget()
                data_view.pack_forget()
                column_view.pack_forget()
            except:
                pass    

            data_view = data_file_view(file_view,df=DATA_TABLE['df'],CONFIG=CONFIG)
            data_view.pack(fill=BOTH,padx=1,pady=1)

            col_explorer = column_explorer(column_view,df=DATA_TABLE['df'])
            col_explorer.pack(side=LEFT)

            return
        
        return    
    def get_item(event): # double clicked    
        #print(">> pressed double click on files list") # monitor
        entry_string = exp_entry.get()
        current_path = '/'.join(entry_string.split('/')[:-1]) + '/'
        file_name = exp_lb.item(exp_lb.selection(),'text')

        exp_entry.delete(0,END)
        exp_entry.insert(0,f"{current_path}{file_name}")
        get_content(event)
        #print(f">> type={file_type}, go_back={file_path.endswith('...')}") # monitor

    frame = ctk.CTkFrame(parent,fg_color='transparent')

    file_label = ctk.CTkLabel(frame,fg_color='transparent',font=(CONFIG['code_block']['font'],CONFIG['code_block']['font_size']))
    if CONFIG['file'] == '':
        file_label.configure(text='>> Pick data file:',text_color=CONFIG['explorer']['cmd_font_color'])

    file_label.pack(side=TOP,fill=X)
    
    exp_entry = ctk.CTkEntry(frame,border_width=1,corner_radius=5,fg_color=CONFIG['code_block']['background'],font=(CONFIG['code_block']['font'],CONFIG['code_block']['font_size']))
    exp_entry.insert(0,path)
    exp_entry.bind('<Return>',get_content) # Bind the return click event to the entry
    exp_entry.pack(side=TOP,pady=2,padx=1,fill=X)

    exp_lb = ttk.Treeview(frame,height=50,show='tree',style="Custom.Treeview")
    exp_lb.column("#0", width=300)  # set treeview width
    update_list_content(list_box=exp_lb,parent_dir=exp_entry.get())

    exp_lb.pack(side=TOP,padx=2,fill=BOTH,expand=True)
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
        data_view.pack_forget()
        data_view = column_view(root,df=df,column=column_name)
        data_view.pack()
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
def data_file_view(parent, df: pd.DataFrame, sample=5, CONFIG=None):
    frame = ctk.CTkScrollableFrame(parent,height=940,fg_color='transparent')

    if DATA_TABLE.get('file_name'):
        file_name = DATA_TABLE['file_name'].split('.')[0]
        new_command(frame, cmd_string=f"{file_name}.shape", data_object='df').pack(side=ctk.TOP, fill=ctk.X, pady=1, padx=1)
        new_command(frame, cmd_string=f"{file_name}.columns", data_object='df').pack(side=ctk.TOP, fill=ctk.X, pady=1, padx=1)
        new_command(frame, cmd_string=f"{file_name}.head({sample})", data_object='df').pack(side=ctk.TOP, fill=ctk.X, pady=1, padx=1)
        new_command(frame, cmd_string=f"{file_name}.tail({sample})", data_object='df').pack(side=ctk.TOP, fill=ctk.X, pady=1, padx=1)
        new_command(frame, cmd_string=f"{file_name}.describe().T", data_object='df').pack(side=ctk.TOP, fill=ctk.X, pady=1, padx=1)
        new_command(frame, cmd_string=f"{file_name}.info()", data_object='df').pack(side=ctk.TOP, fill=ctk.X, pady=1, padx=1)

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
#root = ThemedTk(theme='clearlooks') # equilux/yaru/clearlooks/breeze/arc/adapta/classic/default/scidgreen/scidblue/aqua/breeze-dark/awdark
#root.configure(bg='white')
root = ctk.CTk()
ctk.set_appearance_mode("light")
root.geometry(f"{int(root.winfo_screenwidth()/1.1)}x{int(root.winfo_screenheight()/1.1)}")
root.title('Data Explorer')

style = ttk.Style()
style.configure("Custom.Treeview",
                font=(CONFIG['explorer']['font'],CONFIG['explorer']['font_size']),
                foreground=CONFIG['explorer']['font_color'],  # Text color
                background=CONFIG['explorer']['background'],  # Background color
                fieldbackground=CONFIG['explorer']['selection_color'],  # Field background color (for cells)
                bordercolor=CONFIG['explorer']['frame_color'],
                borderwidth=7,
                rowheight=20)  # Row height
style.configure('Custom.TEntry', 
                foreground=CONFIG['explorer']['font_color'],        # Text color
                background=CONFIG['explorer']['background'],  # Background color
                borderwidth=2,           # Border width
                relief='sunken',         # Border style
                font=(CONFIG['explorer']['font'],CONFIG['explorer']['cmd_font_size']),      # Font
                padding=(1,1,1,1))    # Padding (top, right, bottom, left)

# files panel
left_panel = ctk.CTkFrame(root)
file_explorer = file_explorer(root, path=CONFIG['main_path'])
file_explorer.pack(side=ctk.LEFT)

# board (=right panel)
board = ttk.Notebook(root, padding=15)
board.pack(side=ctk.LEFT, fill="both", expand=True)

# add tabs to board
exp_tab = ttk.Notebook(board, padding=15)
board.add(exp_tab, text='< Explore >>')
story_tab = ttk.Notebook(board, padding=15)
ctk.CTkLabel(story_tab, text='Story', font=(CONFIG['font'], CONFIG['font_size'] + 4)).pack(side=ctk.TOP, pady=20)
board.add(story_tab, text='< Story >>')

file_view = new_tab(exp_tab, text='- File Overview -')
column_view = new_tab(exp_tab, text='- Explore Column -')

root.protocol("WM_DELETE_WINDOW", close_window)
root.mainloop()
