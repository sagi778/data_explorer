from func import *

# constants
CONFIG = load_config('config.json')
'''
CONFIG = {'main_path':"C:/Users/sagic/[2] UFC_Predictor/",#"/mobileye/Perfects/Reports/DATA/argo_pipelines/"
          'file_path': "C:/Users/sagic/[2] UFC_Predictor/data/data_features.csv", #"/mobileye/Perfects/Reports/DATA/argo_pipelines/piplines_archive.csv",
          'supported_files':['csv'],
          'background':'white',
          "entry_color":'#F1F1F1',
          'frame_color':'white',
          'highlight_color':'#F9F9F9',
          'selection_color':'#D5D8DC',
          'highlight_thick':2,
          'table':{'background':'white',
                   'alt_row_color':'#F0F0F0',
                   'font':'Consolas',
                   'font_size':9,
                   'font_color':'#434343',
                   'header_font':'Consolas bold',
                  },
          'data_types_colors':{str:'#B03443',
                               np.int64:'#359973',
                               np.float64:'#733283',
                               },        
          'border':0,
          'border_color':'#CCCCCC',
          'button_frame_color':'#009519',
          'font':'Consolas',
          'font_color':'#434343',
          'font_size':12
          }'''
DATA_TABLE = {'path':CONFIG['file_path'],
              'df':pd.read_csv(CONFIG['file_path'])
             }

# func
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

# basic widgets
def eButton(parent,text:str,CONFIG=CONFIG):
    frame = Frame(parent,bg=CONFIG['button_frame_color'],padx=2,pady=2)
    Button(frame,text=text,bg=CONFIG['entry_color'],bd=CONFIG['border']).pack()
    return frame    

# widgets
def FileExplorer(parent,path:str,CONFIG=CONFIG):
    def get_file_type(file_path:str):
        if file_path.endswith('/'):
            return 'dir'
        else:
            return file_path.split('.')[-1]
    def update_list_content(list_box:Listbox,parent_dir:str): # test
        list_box.delete(0,END)
        for file in ["..."] + get_dir(parent_dir):
                list_box.insert(END,file)
                if get_file_type(file) in CONFIG['supported_files']:
                    list_box.itemconfig(END,{'fg':'green'})

    def get_content(event): # pressed return 
        #print("pressed Return on path entry") # monitor
        path = e.get()
        file_name = path.split('/')[-1]
        file_type = get_file_type(path)

        if file_type == 'dir':
            update_list_content(list_box=l,parent_dir=path)
            return
        
        if file_type in CONFIG['supported_files']: 
            global cl,st,st1
            cl.grid_forget()
            cl = ColumnsList(root,df=pd.read_csv(path))
            cl.grid(row=0,column=1)
            return
        
        return    
    def get_item(event): # double clicked    
        #print(">> pressed double click on files list") # monitor
        file_path = f"{e.get()}{l.get(0,END)[l.curselection()[0]]}"
        file_name = l.get(0,END)[l.curselection()[0]]
        file_type = get_file_type(file_name)

        #print(f">> type={file_type}, go_back={file_path.endswith('...')}") # monitor

        if file_path.endswith('...'):
            parent_dir_path = '/'.join(file_path.split('/')[:-2]) + '/'
            e.delete(0,END)
            e.insert(0,parent_dir_path)
            file_label.config(text='')
            update_list_content(list_box=l,parent_dir='/'.join(file_path.split('/')[:-2])) 
            return
        
        if len(file_path.split('.')) > 2: # bad file path (insert file+file)
            #print(f">> corrupted path={file_path} fixed_path={'/'.join(file_path.split('/')[:-1])}/{file_name}") # monitor
            file_path = f"{'/'.join(file_path.split('/')[:-1])}/{file_name}"
            
        if file_type != 'dir': # item is file
            file_label.config(text=file_name)
            if ~file_path.endswith(file_name):
                e.delete(0,END)
                e.insert(0,file_path)
                #print(f">> {file_type in CONFIG['supported_files']}") # monitor

                if file_type in CONFIG['supported_files']:
                    CONFIG['file_path'] = file_path
                    DATA_TABLE['path'] = CONFIG['file_path']
                    DATA_TABLE['df'] = pd.read_csv(DATA_TABLE['path'])
                    get_content(event)
                else:
                    print(f"File is not supported: {file_name}") 
                return       
        
        # item is folder
        file_label.config(text='')
        e.delete(0,END)
        e.insert(0,file_path)
        update_list_content(list_box=l,parent_dir=file_path)
        return
                   
    frame = Label(parent,bg=CONFIG['background'],padx=2,pady=2)

    file_label = Label(frame,text='',bg=CONFIG['background'],font=(CONFIG['font'],CONFIG['font_size']),width=45,padx=1,pady=0)
    file_label.grid(row=0,column=0)

    entry_frame = Frame(frame,bg=CONFIG['border_color'],padx=1,pady=1)
    e = Entry(entry_frame,width=45,background=CONFIG['entry_color'],fg=CONFIG['font_color'],bd=CONFIG['border'],highlightcolor=CONFIG['highlight_color'],highlightthickness=CONFIG['highlight_thick'],font=(CONFIG['font'],CONFIG['font_size']))
    e.insert(0,path)
    e.bind('<Return>',get_content) # Bind the return click event to the entry
    e.pack()
    entry_frame.grid(row=1,column=0,padx=3)

    lb_frame = Frame(frame,bg=CONFIG['border_color'],padx=1,pady=1)
    l = Listbox(lb_frame,width=45,height=40,activestyle='none',selectbackground=CONFIG['selection_color'],selectforeground='black',fg=CONFIG['font_color'],background=CONFIG['entry_color'],bd=CONFIG['border'],highlightcolor=CONFIG['highlight_color'],highlightthickness=CONFIG['highlight_thick'],font=(CONFIG['font'],CONFIG['font_size']))
    update_list_content(list_box=l,parent_dir=e.get())

    l.pack()
    l.bind('<Double-1>',get_item) # Bind the double click event to the listbox
    lb_frame.grid(row=2,column=0,padx=5,pady=5)
    
    return frame
def ColumnsList(parent,df,CONFIG=CONFIG):
    def update_lb(list_box:Listbox,df:pd.DataFrame): 
        list_box.delete(0,END)

        if df.index.name != None:
            list_box.insert(END, df.index.name)
            list_box.itemconfig(END,{'bg':'#B2D4DA'})

        for column in df.columns.to_list():
            type_string = str(type(df.loc[0,column]))
            data_type = type_string[type_string.find("'",1)+1:type_string.find("'",-1)-1]
            list_box.insert(END,f"({data_type}) {column}")

            # color data types
            if type(df.loc[0,column]) in CONFIG['data_types_colors'].keys():
                list_box.itemconfig(END,{'fg':CONFIG['data_types_colors'][type(df.loc[0,column])]})    

    def get_column(event):
        def get_type(selected_column) -> str:
            type_string = str(type(df.loc[0,selected_column]))
            return type_string[type_string.find("'",1)+1:type_string.find("'",-1)-1]
        def get_column_name(selected_column) -> str:
            return selected_column[selected_column.find(')') + 2:]

        selected_column = lb.get(0,END)[lb.curselection()[0]]
        print(f"Loading data: {get_column_name(selected_column)}")
        column_label.config(text=selected_column)
        global st,st1
        st.grid_forget()
        DATA_TABLE['df'] = pd.DataFrame(df[[get_column_name(selected_column)]])
        st = Sample_DataTable(root,dataframe=DATA_TABLE['df'],sample_size=5)
        st.grid(row=1,column=2)

        st1.grid_forget()
        st1 = Sample_DataTable(root,dataframe=DATA_TABLE['df'].describe(),sample_size=10)
        st1.grid(row=0,column=2)

    frame = Frame(parent,bg=CONFIG['border_color'], padx=1, pady=1)

    column_label = Label(frame,text='',bg=CONFIG['background'],font=(CONFIG['font'],CONFIG['font_size']),width=45,padx=1,pady=0)
    column_label.grid(row=0,column=0,padx=1)
    
    lb = Listbox(frame,width=45,height=40,activestyle='none',selectbackground=CONFIG['selection_color'],selectforeground='black',fg=CONFIG['font_color'],background=CONFIG['entry_color'],bd=CONFIG['border'],highlightcolor=CONFIG['highlight_color'],highlightthickness=CONFIG['highlight_thick'],font=(CONFIG['font'],CONFIG['font_size']))
    update_lb(list_box=lb,df=df)
    lb.grid(row=1,column=0,padx=2)
    lb.bind('<Double-1>',get_column)

    return frame

def Sample_DataTable(parent_frame, dataframe, sample_size=5, table_width=200, table_height=250):
    def column_clicked(column):
        print(f"Button for column {column} clicked")
        column_data.destroy()
        column_data = Sample_DataTable(parent_frame, dataframe[column].describe().to_frame().T, sample_size=5, table_width=table_width, table_height=table_height)
        column_data.grid(row=0, column=1)

    frame = Frame(parent_frame, bg=CONFIG['border_color'], padx=1, pady=1)
    frame.pack_propagate(False)

    if len(dataframe) > sample_size * 2 + 1:
        dataframe_to_show = pd.concat([dataframe.head(sample_size), pd.DataFrame({column: '...' for column in dataframe.columns}, index=['...' + str(sample_size)]), dataframe.tail(sample_size)])
    else:
        dataframe_to_show = dataframe

    x_scrollbar = Scrollbar(frame, orient="horizontal")
    x_scrollbar.grid(row=1, column=0, sticky="ew", columnspan=len(dataframe_to_show.columns) + 1)  # +1 for the index column

    canvas = Canvas(frame, bg=CONFIG['border_color'], highlightthickness=0, xscrollcommand=x_scrollbar.set, width=table_width, height=table_height)
    canvas.grid(row=0, column=0, sticky="nsew")

    x_scrollbar.config(command=canvas.xview)

    inner_frame = Frame(canvas, bg=CONFIG['border_color'])
    canvas.create_window((0, 0), window=inner_frame, anchor="nw")

    # Add index label
    Label(inner_frame, text=dataframe_to_show.index.name, font=(CONFIG['table']['header_font'], CONFIG['table']['font_size'], 'bold'), bd=0, padx=1, pady=1).grid(row=0, column=0, sticky="nsew")

    for col, header in enumerate(dataframe_to_show.columns):
        font_weight = 'bold' if header == dataframe.index.name else 'normal'
        button = Button(inner_frame, text=header, font=(CONFIG['table']['header_font'], CONFIG['table']['font_size'], font_weight), bd=0, padx=1, pady=1, command=lambda col=header: column_clicked(header))
        button.grid(row=0, column=col + 1, sticky="nsew")  # +1 to account for index column

    for row_index, (index, row_data) in enumerate(dataframe_to_show.iterrows()):
        bg_color = CONFIG['table']['background']
        Label(inner_frame, text=index, font=(CONFIG['font'], CONFIG['font_size']), fg=CONFIG['table']['font_color'], padx=1, pady=1, bg=bg_color).grid(row=row_index + 1, column=0, sticky="nsew")
        for col, value in enumerate(row_data):
            Label(inner_frame, text=value, font=(CONFIG['font'], CONFIG['font_size']), fg=CONFIG['table']['font_color'], padx=1, pady=1, bg=bg_color).grid(row=row_index + 1, column=col + 1, sticky="nsew")  # +1 to account for index column

    inner_frame.update_idletasks()
    canvas.config(scrollregion=canvas.bbox("all"))

    for i in range(dataframe_to_show.shape[1] + 1):  # +1 for the index column
        inner_frame.columnconfigure(i, weight=1)
    for i in range(dataframe_to_show.shape[0] + 1):
        inner_frame.rowconfigure(i, weight=1)

    return frame
 

# main
root = Tk()
root.configure(bg='white')
root.geometry("1300x900")
root.title('Explorer')

FileExplorer(root,CONFIG['main_path']).grid(row=0,column=0,rowspan=2,padx=1,pady=1)

cl = ColumnsList(root,DATA_TABLE['df'])
cl.grid(row=0,column=1,rowspan=2,padx=3,pady=3)

st = Sample_DataTable(root,dataframe=DATA_TABLE['df'],sample_size=5)
st.grid(row=0,column=2,padx=1)

st1 = Sample_DataTable(root,dataframe=pd.DataFrame(),table_width=400)
st1.grid(row=1,column=2,padx=1)

root.mainloop()