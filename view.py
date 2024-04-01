from tkinter import *
import os
import pandas as pd

# constants
CONFIG = {'main_path':"C:/Users/sagic/[5] Net_Worth/",
          'background':'white',
          "entry_color":'#F1F1F1',
          'frame_color':'white',
          'highlight_color':'#43AFFF',
          'highlight_thick':2,
          'border':0,
          'border_color':'#CCCCCC',
          'button_frame_color':'#009519',
          'font':'Consolas',
          'font_size':11
          }

# func
def listbox_double_click(event):
    w = event.widget
    index = w.curselection()[0]
    selected_item = w.get(index)
    print("Double click on item:", selected_item) # monitor
def get_content(directory):
    try:
        contents = os.listdir(directory)
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
    frame = Frame(parent,bg=CONFIG['border_color'],padx=2,pady=2)
    Label(frame,text=file_path).pack()
    return frame
def FileExplorer(parent,path:str,CONFIG=CONFIG):
    
    def get_dir_content(entry:Entry,listbox:Listbox):
        #print(f"{get_content(e.get())}") # monitor
        listbox.delete(0,END)
        for file in [". . ."] + get_content(e.get()):
            listbox.insert(END,file)
    def get_file(entry:Entry,listbox:Listbox):
        file_path = f"{e.get()}{listbox.get(0,END)[listbox.curselection()[0]]}"
        print(f"loading {file_path}:") # monitor
        try:
            df = pd.read_csv(file_path)
            print(df.tail())
        except:
            print(f"{file_path} is not data file.")    

    frame = Label(parent,bg=CONFIG['background'],padx=2,pady=2)
    entry_frame = Frame(frame,bg=CONFIG['border_color'],padx=1,pady=1)
    e = Entry(entry_frame,width=45,background=CONFIG['entry_color'],bd=CONFIG['border'],highlightcolor=CONFIG['highlight_color'],highlightthickness=CONFIG['highlight_thick'],font=(CONFIG['font'],CONFIG['font_size']))
    e.insert(0,path)
    e.pack()
    entry_frame.grid(row=0,column=0,padx=3)

    chosen_file = StringVar()
    chosen_file.set(get_content(path)[0])
    lb_frame = Frame(frame,bg=CONFIG['border_color'],padx=1,pady=1)
    l = Listbox(lb_frame,width=45,height=20,background=CONFIG['entry_color'],bd=CONFIG['border'],highlightcolor=CONFIG['highlight_color'],highlightthickness=CONFIG['highlight_thick'],font=(CONFIG['font'],CONFIG['font_size']))
    for file in ["..."] + get_content(e.get()):
        l.insert(END,file)

    l.pack()
    l.bind('<Double-1>',listbox_double_click) # Bind the double click event to the listbox
    lb_frame.grid(row=1,column=0,padx=5,pady=5)

    b_entry = Button(frame,text=">",bd=CONFIG['border'],padx=10,pady=1,command=lambda:get_dir_content(entry=e,listbox=l))
    b_entry.grid(row=0,column=1)
    
    return frame

# main
root = Tk()
root.configure(bg='white')
root.geometry("800x800")
root.title('EDA tool')

#FileStatus(root,"C:/Users/sagic/[5] Net_Worth/VIX_Tracker.csv").grid(row=0,column=0,padx=3,pady=3) # need to check
FileStatus(root,file_path=f"{CONFIG['main_path']}").grid(row=0,column=0)
FileExplorer(root,CONFIG['main_path']).grid(row=1,column=0,padx=3,pady=3)


root.mainloop()