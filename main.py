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
root.geometry(f"{int(root.winfo_screenwidth()/1.1)}x{int(root.winfo_screenheight()/1.1)}")
root.title('Data Explorer')

# tabview panel (=right panel)
tv = Tabview(root)
tv.pack(side=RIGHT,fill=BOTH,expand=True)

tv.add_tab('<Exp File>')
exp_file_tab = tv.board.tab('<Exp File>')
tv.add_tab('<Exp Column>')
exp_column_tab = tv.board.tab('<Exp Column>')
tv.add_tab('<Story>')
story_tab = tv.board.tab('<Story>')
story_frame = ctk.CTkScrollableFrame(story_tab,fg_color='white',bg_color='white')
story_frame.pack(fill=BOTH,expand=True)

# views
fv = FileView(exp_file_tab)
fv.pack(fill=BOTH)

cv = ColumnView(exp_column_tab)
cv.pack(fill=BOTH)

# file explorer
file_exp = FileExplorer(master=root,path=CONFIG['main_path'],file_view_frame=fv,column_view_frame=cv)
file_exp.pack(side=RIGHT)

# r&d widget
#TextOutput(exp_file_tab,out_string='dsfgdsfg').pack(side=TOP,expand=False)

root.protocol("WM_DELETE_WINDOW", close_window)
root.mainloop()