from tkinter import *
import os

# constants
MAIN_PATH = "/mobileye/Perfects/Reports/DATA/argo_pipelines/"

# func
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

# widgets
def FileExplorer(parent,path:str):
    ENT_COLOR = '#F3F3F3'
    FRAME_COLOR = 'white'
    HIGHLIGHT_COLOR = '#43AFFF'
    BORDER = 0
    
    def get_dir_content(entry:Entry,listbox:Listbox):
        #print(f"{get_content(e.get())}") # monitor
        listbox.delete(0,END)
        for file in [". . ."] + get_content(e.get()):
            listbox.insert(END,file)

    def get_file(entry:Entry,listbox:Listbox):
        file_path = f"{e.get()}{listbox.get(0,END)[listbox.curselection()[0]]}"
        print(f"loading {file_path}")

    frame = LabelFrame(parent,text='File Explorer',bg=FRAME_COLOR)
    e = Entry(frame,width=45,background=ENT_COLOR,bd=BORDER,highlightcolor=HIGHLIGHT_COLOR)
    e.insert(0,path)
    e.grid(row=0,column=0,padx=3)

    chosen_file = StringVar()
    chosen_file.set(get_content(path)[0])
    l = Listbox(frame,width=45,height=20,background=ENT_COLOR,bd=BORDER,highlightcolor=HIGHLIGHT_COLOR)
    for file in [". . ."] + get_content(e.get()):
        l.insert(END,file)

    l.grid(row=1,column=0,padx=5,pady=5)

    b_entry = Button(frame,text=">",padx=10,pady=1,command=lambda:get_dir_content(entry=e,listbox=l))
    b_entry.grid(row=0,column=1)

    b_list = Button(frame,text=">",padx=3,pady=3,command=lambda:get_file(entry=e,listbox=l))
    b_list.grid(row=1,column=2)
    
    return frame

# main
root = Tk()
root.configure(bg='white')
root.geometry("800x800")
root.title('EDA tool')

FileExplorer(root,MAIN_PATH).grid(row=0,column=0,padx=3,pady=3)
LabelFrame(root,text='Data Frame').grid(row=0,column=1)

root.mainloop()