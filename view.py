from tkinter import *
from tkinter import ttk
import os
import pandas as pd
from pandastable import Table, TableModel

# constants
CONFIG = {'main_path':"/mobileye/Perfects/Reports/DATA/argo_pipelines/", #"C:/Users/sagic/[5] Net_Worth/",
          'file_path':"/mobileye/Perfects/Reports/DATA/argo_pipelines/piplines_archive.csv",
          'supported_files':['csv'],
          'background':'white',
          "entry_color":'#F1F1F1',
          'frame_color':'white',
          'highlight_color':'#43AFFF',
          'highlight_thick':2,
          'border':0,
          'border_color':'#CCCCCC',
          'button_frame_color':'#009519',
          'font':'Consolas',
          'font_color':'#434343',
          'font_size':11
          }

# func
def get_content(directory):
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
        for file in ["..."] + get_content(parent_dir):
                list_box.insert(END,file)
                if get_file_type(file) in CONFIG['supported_files']:
                    list_box.itemconfig(END,{'fg':'green'})
            
    def get_dir_content(event): # pressed return 
        #print("pressed Return on path entry") # monitor
        path = e.get()
        file_name = path.split('/')[-1]
        file_type = get_file_type(path)

        if file_type == 'dir':
            update_list_content(list_box=l,parent_dir=path)
            return
        
        if file_type in CONFIG['supported_files']: 
            global dt
            dt.grid_forget()
            dt = Sample_DataTable(root,df=pd.read_csv(path),sample_size=5)
            dt.grid(row=0,column=1)
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
    entry_frame = Frame(frame,bg=CONFIG['border_color'],padx=1,pady=1)
    e = Entry(entry_frame,width=45,background=CONFIG['entry_color'],fg=CONFIG['font_color'],bd=CONFIG['border'],highlightcolor=CONFIG['highlight_color'],highlightthickness=CONFIG['highlight_thick'],font=(CONFIG['font'],CONFIG['font_size']))
    e.insert(0,path)
    e.bind('<Return>',get_dir_content) # Bind the return click event to the entry
    e.pack()
    entry_frame.grid(row=0,column=0,padx=3)

    file_label = Label(frame,text='',bg=CONFIG['background'],font=(CONFIG['font'],CONFIG['font_size']),width=45,padx=1,pady=1)
    file_label.grid(row=1,column=0)

    lb_frame = Frame(frame,bg=CONFIG['border_color'],padx=1,pady=1)
    l = Listbox(lb_frame,width=45,height=50,fg=CONFIG['font_color'],background=CONFIG['entry_color'],bd=CONFIG['border'],highlightcolor=CONFIG['highlight_color'],highlightthickness=CONFIG['highlight_thick'],font=(CONFIG['font'],CONFIG['font_size']))
    update_list_content(list_box=l,parent_dir=e.get())

    l.pack()
    l.bind('<Double-1>',get_item) # Bind the double click event to the listbox
    lb_frame.grid(row=2,column=0,padx=5,pady=5)
    
    return frame
def Sample_DataTable(parent,df,sample_size:int = 5):
    frame = Frame()
    frame.pack_propagate(False)  # Prevent frame from resizing to fit its contents
    
    # Slice DataFrame to show only the first and last 5 rows
    df_to_show = pd.concat([df.head(sample_size),pd.DataFrame({column:'...' for column in df.columns},[sample_size]),df.tail(sample_size)])

    # Create column headers
    for col, header in enumerate(df_to_show.columns):
        label = Label(frame, text=header, padx=10, pady=5)
        label.grid(row=0, column=col, sticky="nsew")

    # Insert data rows
    for row_index, row_data in df_to_show.iterrows():
        bg_color = CONFIG['entry_color'] if row_index % 2 == 0 else CONFIG['frame_color']
        for col, value in enumerate(row_data):
            label = Label(frame, text=value, padx=10, pady=5, bg=bg_color)
            label.grid(row=row_index + 1, column=col, sticky="nsew")

    # Configure row and column weights to fill the entire space
    for i in range(df_to_show.shape[1]):
        frame.columnconfigure(i, weight=1)
    for i in range(df_to_show.shape[0] + 1):
        frame.rowconfigure(i, weight=1)

    return frame

 
# main
root = Tk()
root.configure(bg='white')
root.geometry("1500x900")
root.title('Explorer')

FileExplorer(root,CONFIG['main_path']).grid(row=0,column=0,padx=3,pady=3)

df = pd.read_csv(CONFIG['file_path'])
#df = pd.DataFrame()
dt = Sample_DataTable(root,df,sample_size=5)
dt.grid(row=0,column=1)

root.mainloop()