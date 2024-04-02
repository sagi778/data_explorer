from tkinter import *
import os
import pandas as pd

# constants
CONFIG = {'main_path':"/mobileye/Perfects/Reports/DATA/argo_pipelines/", #"C:/Users/sagic/[5] Net_Worth/",
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
def FileStatus(parent,file_path:str,CONFIG=CONFIG):
    frame = Frame(parent,bg=CONFIG['border_color'],padx=2,pady=2,width=50)
    Label(frame,text=file_path).pack()
    return frame
def FileExplorer(parent,path:str,CONFIG=CONFIG):
    def get_file_type(file_path:str):
        try:
            file_type = file_path.split('.')[file_path.count('.')]
            return file_type
        except:
            return 'dir'
    def get_dir_content(event): # return 
        path = e.get()
        l.delete(0,END)
        for file in ["..."] + get_content(path):
            l.insert(END,file)
    def get_item(event): # double clicked    
        file_path = f"{e.get()}{l.get(0,END)[l.curselection()[0]]}"
        e.delete(0,END)
        e.insert(0,file_path)

        if get_file_type(file_path) in ['csv']:
            e.delete(0,END)
            e.insert(0,file_path)
            df = pd.read_csv(file_path)
        elif '...' in file_path: # go up 1 dir
            new_path = '/'.join(file_path.split('/')[:-1]) + ['/']
            e.delete(0,END)
            e.insert(0,new_path)
            l.delete(0,END)
            for file in ["..."] + get_content(new_path):
                l.insert(END,file)
        else: # go in 1 dir    
            l.delete(0,END)
            for file in ["..."] + get_content(file_path):
                l.insert(END,file)
            
    frame = Label(parent,bg=CONFIG['background'],padx=2,pady=2)
    entry_frame = Frame(frame,bg=CONFIG['border_color'],padx=1,pady=1)
    e = Entry(entry_frame,width=45,background=CONFIG['entry_color'],fg=CONFIG['font_color'],bd=CONFIG['border'],highlightcolor=CONFIG['highlight_color'],highlightthickness=CONFIG['highlight_thick'],font=(CONFIG['font'],CONFIG['font_size']))
    e.insert(0,path)
    e.bind('<Return>',get_dir_content) # Bind the return click event to the entry
    e.pack()
    entry_frame.grid(row=0,column=0,padx=3)

    chosen_file = StringVar()
    chosen_file.set(get_content(path)[0])
    lb_frame = Frame(frame,bg=CONFIG['border_color'],padx=1,pady=1)
    l = Listbox(lb_frame,width=45,height=100,fg=CONFIG['font_color'],background=CONFIG['entry_color'],bd=CONFIG['border'],highlightcolor=CONFIG['highlight_color'],highlightthickness=CONFIG['highlight_thick'],font=(CONFIG['font'],CONFIG['font_size']))
    for file in ["..."] + get_content(e.get()):
        l.insert(END,file)

    l.pack()
    l.bind('<Double-1>',get_item) # Bind the double click event to the listbox
    lb_frame.grid(row=1,column=0,padx=5,pady=5)
    
    return frame

# main
root = Tk()
root.configure(bg='white')
root.geometry("800x800")
root.title('EDA tool')

FileExplorer(root,CONFIG['main_path']).grid(row=0,column=0,padx=3,pady=3)
FileStatus(root,file_path=f"{CONFIG['main_path']}").grid(row=1,column=0)

root.mainloop()