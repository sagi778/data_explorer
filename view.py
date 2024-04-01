from tkinter import *
import os

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
    ENT_COLOR = '#E1E1E1'
    FRAME_COLOR = 'white'
    def get_selected_path(path):
        print(path)

    frame = LabelFrame(parent,text='File Explorer',bg=FRAME_COLOR)
    e = Entry(frame,width=40,background=ENT_COLOR)
    e.insert(0,path)
    e.grid(row=0,column=0,padx=3)

    chosen_file = StringVar()
    chosen_file.set(get_content(path)[0])
    l = Listbox(frame,width=40,height=20,background=ENT_COLOR)
    for file in [". . ."] + get_content(path):
        l.insert(END,file)
    l.grid(row=1,column=0,columnspan=2,padx=5,pady=5)

    b = Button(parent,text=">",command=lambda:get_selected_path(f"{path}{get_content(path)[l.curselection()[0]]}"))
    b.grid(row=0,column=1)

    
    return frame

root = Tk()
root.configure(bg='white')
root.geometry("800x800")
root.title('EDA tool')

FileExplorer(root,'C:/Users/sagic/[5] Net_Worth/').grid(row=0,column=0,padx=10,pady=10)

root.mainloop()