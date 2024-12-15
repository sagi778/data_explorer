from func import *
from widgets import *

# load data
DATA_TABLE = {'path':CONFIG['file'],
              'file_name':None,
              'df':pd.DataFrame() if CONFIG['file']=='' else pd.read_csv(CONFIG['file'])
             }

# operation func
def close_window():
    if messagebox.askyesno("Quit", "Do you want to quit?"):
        print('bye bye...')
        root.quit()
        root.destroy()   

# main
root = ctk.CTk()
root.configure(bg_color=CONFIG['background'])
ctk.set_appearance_mode("light")
WIND_HEIGHT,WIND_WIDTH = int(root.winfo_screenheight()/1.1),int(root.winfo_screenwidth()/1.1)
root.geometry(f"{WIND_WIDTH}x{WIND_HEIGHT}")
root.title('Data Explorer')

# views panel
data_explorer = ctk.CTkScrollableFrame(
    master=root,
    bg_color='transparent',
    fg_color='transparent',
    scrollbar_fg_color='transparent',
    scrollbar_button_color='white',
    width=WIND_WIDTH-200,height=WIND_HEIGHT,
    )
data_explorer.pack(side=RIGHT,expand=True,pady=2,padx=2)

# file explorer panel
file_explorer = FileExplorer(
    master=root,
    data_explorer=data_explorer,
    path=CONFIG['main_path'])
file_explorer.pack(side=LEFT,pady=2,padx=2)

root.protocol("WM_DELETE_WINDOW", close_window)
root.mainloop()

