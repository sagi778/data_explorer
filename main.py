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
            file_label.config(text=file_name)
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

    frame = Label(parent,bg=CONFIG['background'],padx=2,pady=2)

    file_label = Label(frame,text=CONFIG['file'],bg=CONFIG['background'],font=(CONFIG['font'],CONFIG['font_size']),width=45,padx=1,pady=0)
    file_label.grid(row=0,column=0)
    
    entry_frame = Frame(frame,bg=CONFIG['border_color'],padx=1,pady=1)
    exp_entry = Entry(entry_frame,width=45,background=CONFIG['entry_color'],fg=CONFIG['font_color'],bd=CONFIG['border'],highlightcolor=CONFIG['highlight_color'],highlightthickness=CONFIG['highlight_thick'],font=(CONFIG['font'],CONFIG['font_size']))
    exp_entry.insert(0,path)
    exp_entry.bind('<Return>',get_content) # Bind the return click event to the entry
    exp_entry.pack()
    entry_frame.grid(row=1,column=0,padx=3)

    lb_frame = Frame(frame,bg=CONFIG['border_color'],padx=1,pady=1)
    exp_lb = Listbox(lb_frame,width=45,height=40,activestyle='none',selectbackground=CONFIG['selection_color'],selectforeground='black',fg=CONFIG['font_color'],background=CONFIG['entry_color'],bd=CONFIG['border'],highlightcolor=CONFIG['highlight_color'],highlightthickness=CONFIG['highlight_thick'],font=(CONFIG['font'],CONFIG['font_size']))
    update_list_content(list_box=exp_lb,parent_dir=exp_entry.get())

    exp_lb.pack()
    exp_lb.bind('<Double-1>',get_item) # Bind the double click event to the listbox
    lb_frame.grid(row=2,column=0,padx=5,pady=5)

    return frame
def column_explorer(parent,df:pd.DataFrame,CONFIG=CONFIG):
    def update_list_content(list_box:Listbox,df:pd.DataFrame):
        list_box.delete(0,END)
        for column in df.columns.tolist():
            data_type = str(df[column].dtype)
            list_box.insert(END,f"({data_type}) {column}")
            if data_type.startswith('int'):
                list_box.itemconfig(END,{'fg':'#239B56'})
            if data_type.startswith('float'):
                list_box.itemconfig(END,{'fg':'#1F618D'})
            if data_type.startswith('object'):
                list_box.itemconfig(END,{'fg':'#6C3483'})        
         
    def get_content(event): # pressed return 
        print(f"pressed Return")        
    def get_column(event): # double clicked    
        print(f'Loading {exp_lb.get(0,END)[exp_lb.curselection()[0]]}')
        column_name = exp_lb.get(0,END)[exp_lb.curselection()[0]]
        exp_entry.delete(0,END)
        exp_entry.insert(0,column_name)

    frame = Label(parent,bg=CONFIG['background'],padx=2,pady=2)

    file_label = Label(frame,text=f"{DATA_TABLE['file_name']}.colummns",bg=CONFIG['background'],font=(CONFIG['font'],CONFIG['font_size']),width=45,padx=1,pady=0)
    file_label.grid(row=0,column=0)
    
    entry_frame = Frame(frame,bg=CONFIG['border_color'],padx=1,pady=1)
    exp_entry = Entry(entry_frame,width=45,background=CONFIG['entry_color'],fg=CONFIG['font_color'],bd=CONFIG['border'],highlightcolor=CONFIG['highlight_color'],highlightthickness=CONFIG['highlight_thick'],font=(CONFIG['font'],CONFIG['font_size']))
    exp_entry.insert(0,'')
    exp_entry.bind('<Return>',get_content) # Bind the return click event to the entry
    exp_entry.pack()
    entry_frame.grid(row=1,column=0,padx=3)

    lb_frame = Frame(frame,bg=CONFIG['border_color'],padx=1,pady=1)
    exp_lb = Listbox(lb_frame,width=45,height=40,activestyle='none',selectbackground=CONFIG['selection_color'],selectforeground='black',fg=CONFIG['font_color'],background=CONFIG['entry_color'],bd=CONFIG['border'],highlightcolor=CONFIG['highlight_color'],highlightthickness=CONFIG['highlight_thick'],font=(CONFIG['font'],CONFIG['font_size']))
    update_list_content(list_box=exp_lb,df=df)

    exp_lb.pack()
    exp_lb.bind('<Double-1>',get_column) # Bind the double click event to the listbox
    lb_frame.grid(row=2,column=0,padx=5,pady=5)

    return frame


# main
root = Tk()
root.configure(bg='white')
root.geometry(f"{int(root.winfo_screenwidth()/2)}x{int(root.winfo_screenheight()/2)}")
root.title('Explorer')

file_explorer = file_explorer(root,path=CONFIG['main_path'])
file_explorer.grid(row=0,column=0)

col_explorer = column_explorer(root,df=DATA_TABLE['df'])
col_explorer.grid(row=0,column=1)

root.mainloop()