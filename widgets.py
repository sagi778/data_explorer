from func import *


# load config files
CURRENT_PATH = '\\'.join(os.path.abspath(__file__).split('\\')[:-1]) + '\\'
CONFIG = load_json(f'{CURRENT_PATH}config.json')
COMMANDS = load_json(f'{CURRENT_PATH}commands.json')

# load data
DATA_TABLE = {'path':CONFIG['file'],
              'file_name':None,
              'df':pd.DataFrame() if CONFIG['file']=='' else read_data_file(CONFIG['file'])
             }

# basic widgets
class Entry(ctk.CTkFrame):
    def __init__(self, master=None, width:int=50, **kwargs):
        super().__init__(master, **kwargs)

        self.entry = ctk.CTkEntry(self,
                          width=width,
                          border_width=1,
                          border_color=get_darker_color(CONFIG['code_block']['background'],percentage=30),
                          corner_radius=1,
                          text_color=CONFIG['code_block']['font_color'],
                          fg_color=CONFIG['code_block']['background'],
                          font=(CONFIG['code_block']['font'],CONFIG['code_block']['font_size']+4)
                          )  
        self.entry.pack(padx=1,pady=1)

    def get(self):
        return self.entry.get()
    def set(self,string:str=''):
        self.entry.delete(0,END)
        self.entry.insert(0,string)   
class CodeLine(ctk.CTkFrame):
    def __init__(self, master=None, width:int=500, id:int=0, **kwargs):
        super().__init__(master, **kwargs)

        self._id = id
        self.configure(corner_radius=0,bg_color='transparent',fg_color='transparent',border_width=0)

        self.id = Text(self,
                       height=1,
                       width=3,
                       bd=0,
                       fg=get_darker_color(CONFIG['code_block']['background'],percentage=50),
                       font=(CONFIG['code_block']['font'],CONFIG['code_block']['font_size'])
                       )
        self.id.insert("1.0",f'[{self._id}]')
        self.id.configure(state='disabled')
        self.id.pack(side=LEFT,expand=True,padx=1,pady=1)

        self.frame = Frame(self,
                           width=width,
                           highlightthickness=1,
                           bg=get_darker_color(CONFIG['code_block']['background'],percentage=30),
                           highlightcolor=CONFIG['code_block']['highlight_color'],
                           bd=1
                           )
        self.frame.pack() 

        self.entry = Text(self.frame,
                          width=width,
                          height=1,
                          bg=CONFIG['code_block']['background'],
                          font=(CONFIG['code_block']['font'],CONFIG['code_block']['font_size']),
                          fg=CONFIG['code_block']['font_color'],
                          bd=0,
                          )
        self.entry.pack(side=LEFT,fill=BOTH,expand=False,padx=0,pady=0)

        # binds
        self.entry.bind("<MouseWheel>", self.prevent_mouse_wheel)

    def prevent_mouse_wheel(self,event):
        return 'break'
    def get(self):
        return self.entry.get("1.0","end-1c")
    def get_code(self):
        if '>>' in self.get():
            cmd = self.get()
            return cmd[cmd.find('>> ')+len('>> '):]   
        return self.get()     
    def set(self,string:str=''):
        self.entry.delete("1.0",END)
        self.entry.insert("1.0",string)    
    def set_colors(self):
        def set_run(text,entry=self.entry):
            entry.tag_add("run","1.0","1.3") 
        def set_digits(text, entry=self.entry):
            for i,char in enumerate(text):
                if char.isdigit():
                    start_index = f"1.0 + {i} chars"
                    end_index = f"1.0 + {i+1} chars"
                    entry.tag_add('digit',start_index,end_index)
        def set_string(text,entry=self.entry):
            marker = None
            for i,char in enumerate(text):
                if marker!=None and char==marker:
                    entry.tag_add('string',f"1.0 + {i} chars",f"1.0 + {i+1} chars")
                    marker = None
                elif marker!=None:
                    entry.tag_add('string',f"1.0 + {i} chars",f"1.0 + {i+1} chars")
                elif char=="'" or char=='"':
                    marker = char
                    entry.tag_add('string',f"1.0 + {i} chars",f"1.0 + {i+1} chars")
        def set_functions(text, entry=self.entry):
            start_index = None
            end_index = None
            for i,char in enumerate(text): # run from end to begining
                if char in ['(','[','\n']:
                    end_index = f"1.0 + {i} chars"
                if char in ['.']:
                    start_index = f"1.0 + {i+1} chars"    
            
            if start_index!=None and end_index!=None:
                entry.tag_add('function',start_index,end_index)
        def set_df(text,entry=self.entry):
            if 'df' in text:
                for i,char in enumerate(text):
                    if char == 'd' and text[i+1] == 'f' and text[i+2] == '.':
                        start_index = f"1.0 + {i} chars"  
                        end_index = f"1.0 + {i+len('df.')} chars" 
                        entry.tag_add('df',start_index,end_index)
                        return

        text = self.get()

        set_run(text,entry=self.entry)
        set_digits(text,entry=self.entry)
        set_functions(text,entry=self.entry)
        set_df(text,entry=self.entry)
        set_string(text,entry=self.entry)

        for tag in CONFIG['code_block']['code_tags'].keys():
            self.entry.tag_configure(tag,foreground=CONFIG['code_block']['code_tags'][tag],font=(CONFIG['code_block']['font'],CONFIG['code_block']['font_size'],'bold'))                   
class CodeControls(ctk.CTkFrame):
    def __init__(self, master=None, **kwargs):
        super().__init__(master, **kwargs)

        run_icon = ctk.CTkImage(Image.open(f"{CURRENT_PATH}\\icons\\play.png"))
        save_icon = ctk.CTkImage(Image.open(f"{CURRENT_PATH}\\icons\\thumbtack.png"))
        delete_icon = ctk.CTkImage(Image.open(f"{CURRENT_PATH}\\icons\\trash.png"))
        add_icon = ctk.CTkImage(Image.open(f"{CURRENT_PATH}\\icons\\add.png"))
        up_icon = ctk.CTkImage(Image.open(f"{CURRENT_PATH}\\icons\\arrow-up.png"))
        down_icon = ctk.CTkImage(Image.open(f"{CURRENT_PATH}\\icons\\arrow-down.png"))

        RADIUS = 100

        self.configure(bg_color='transparent',border_width=0,corner_radius=0,fg_color='transparent')

        self.run = ctk.CTkButton(self,
                                 width=10,
                                 height=10,
                                 image=run_icon,
                                 corner_radius=RADIUS,
                                 text='',
                                 fg_color='transparent',
                                 border_width=0,
                                 hover_color=CONFIG['code_block']['background']
                                 )
        self.run.pack(side=LEFT,expand=False,pady=2,padx=2)    

        self.save = ctk.CTkButton(self,
                                 width=10,
                                 height=10,
                                 image=save_icon,
                                 corner_radius=RADIUS,
                                 text='',
                                 fg_color='transparent',
                                 border_width=0,
                                 hover_color=CONFIG['code_block']['background']
                                 )
        self.save.pack(side=LEFT,expand=False,pady=2,padx=2)   
        
        self.delete = ctk.CTkButton(self,
                                 width=10,
                                 height=10,
                                 image=delete_icon,
                                 corner_radius=RADIUS,
                                 text='',
                                 fg_color='transparent',
                                 border_width=0,
                                 border_spacing=1,
                                 hover_color=CONFIG['code_block']['background'],
                                 border_color= get_darker_color(CONFIG['code_block']['background'],percentage=10)
                                 )
        self.delete.pack(side=LEFT,expand=False,pady=2,padx=2) 

        self.up = ctk.CTkButton(self,
                                 width=10,
                                 height=10,
                                 image=up_icon,
                                 corner_radius=RADIUS,
                                 text='',
                                 fg_color='transparent',
                                 border_width=0,
                                 border_spacing=1,
                                 hover_color=CONFIG['code_block']['background'],
                                 border_color= get_darker_color(CONFIG['code_block']['background'],percentage=10)
                                 )
        self.up.pack(side=LEFT,expand=False,pady=2,padx=2)

        self.down = ctk.CTkButton(self,
                                 width=10,
                                 height=10,
                                 image=down_icon,
                                 corner_radius=RADIUS,
                                 text='',
                                 fg_color='transparent',
                                 border_width=0,
                                 border_spacing=1,
                                 hover_color=CONFIG['code_block']['background'],
                                 border_color= get_darker_color(CONFIG['code_block']['background'],percentage=10)
                                 )
        self.down.pack(side=LEFT,expand=False,pady=2,padx=2)

        self.progress_bar = ttk.Progressbar(self,orient=HORIZONTAL,length=100) # need to fix progress bar
        self.progress_bar.pack(side=LEFT,padx=5,pady=3)
        
        '''
        self.add = ctk.CTkButton(self,
                                 width=10,
                                 height=10,
                                 image=add_icon,
                                 corner_radius=RADIUS,
                                 text='',
                                 fg_color='transparent',
                                 border_width=0,
                                 border_spacing=1,
                                 hover_color=CONFIG['code_block']['background'],
                                 border_color= get_darker_color(CONFIG['code_block']['background'],percentage=10)
                                 )
        self.add.pack(side=LEFT,expand=False,pady=2,padx=2)
        '''
          
class Tabview(ctk.CTkFrame):
    def __init__(self, master=None, **kwargs):
        super().__init__(master, **kwargs)

        self.board = ctk.CTkTabview(self,
                                corner_radius=5,
                                border_width=1,
                                text_color='black',
                                bg_color='transparent',
                                fg_color= 'white',
                                segmented_button_fg_color=get_darker_color(CONFIG['tabs']['frame_color'],percentage=30),
                                segmented_button_selected_color='white',
                                segmented_button_selected_hover_color='white',
                                segmented_button_unselected_hover_color=get_darker_color(CONFIG['tabs']['hover_color'],percentage=20),
                                segmented_button_unselected_color=get_darker_color(CONFIG['tabs']['hover_color'],percentage=10)
                                )    
        self.board._segmented_button.configure(font=(CONFIG['code_block']['font'],16,'normal'))
        self.board.pack(fill=BOTH,expand=True,padx=1,pady=1)

    def add_tab(self, name:str=''):
        self.board.add(name)                            

# output widgets
class TextOutput(Frame):
    def __init__(self, master=None, text:str='',width=80, **kwargs):
        super().__init__(master, **kwargs)

        self._text = text

        self.tb = Text(self,
                       wrap='word',
                       height= 1 + self._text.count('\n'),
                       width=width,
                       bd=0,
                       fg=CONFIG['table']['font_color'],
                       font=(CONFIG['table']['font'],CONFIG['table']['font_size']-2)
                       )
        self.tb.pack(side=TOP,expand=True,fill=BOTH)   

        # Create a horizontal Scrollbar
        self.h_scroll = ttk.Scrollbar(self, orient='horizontal', command=self.tb.xview)
        self.tb.configure(xscrollcommand=self.h_scroll.set)

        self.set(self._text) 
      
    def set(self,text:str):
        def set_colors(text,tb):
            def set_digits(text,tb):
                for i,char in enumerate(text):
                    if char.isdigit():
                        tb.tag_add('digit',f"1.0 + {i} chars",f"1.0 + {i+1} chars")
            def set_string(text,tb):
                marker = None
                for i,char in enumerate(text):
                    if marker!=None and char==marker:
                        tb.tag_add('string',f"1.0 + {i} chars",f"1.0 + {i+1} chars")
                        marker = None
                    elif marker!=None:
                        tb.tag_add('string',f"1.0 + {i} chars",f"1.0 + {i+1} chars")
                    elif char=="'" or char=='"':
                        marker = char
                        tb.tag_add('string',f"1.0 + {i} chars",f"1.0 + {i+1} chars")
                   
            # add colors:
            set_digits(text,tb=self.tb)
            set_string(text,tb=self.tb)

            for tag in CONFIG['code_block']['code_tags'].keys():
                self.tb.tag_configure(tag,foreground=CONFIG['code_block']['code_tags'][tag],font=(CONFIG['code_block']['font'],CONFIG['code_block']['font_size'],'bold'))   
        
        self.tb.configure(state='normal')
        self.tb.delete("1.0", END)
        self.tb.insert("1.0",text)
        set_colors(text=text,tb=self)
        self.tb.configure(state='disabled')
class TableOutput(Frame):
    def __init__(self, master=None, df:pd.DataFrame=pd.DataFrame(), **kwargs):
        super().__init__(master, **kwargs)

        self._df = df
        self._length = self.get_length()
        self._width = self.get_width()

        self.tb = Text(self,
                       wrap='none',
                       height= self._length,
                       bd=0,
                       fg=CONFIG['table']['font_color'],
                       font=(CONFIG['table']['font'],CONFIG['table']['font_size'])
                       )
        self.tb.pack(side=TOP,expand=True,fill=BOTH)   

        # Create a horizontal Scrollbar
        self.h_scroll = ttk.Scrollbar(self, orient='horizontal', command=self.tb.xview)
        self.tb.configure(xscrollcommand=self.h_scroll.set)

        self.set(self._df) 
        self.pack_hscrollbar()
        
    
    def get(self):
        return self.tb.get("1.0","end")
    def pack_hscrollbar(self):
        """Check if the horizontal scrollbar is necessary and show/hide it accordingly"""
        widget_width = self.tb['width']
        #print(f"text_width={self._width} ?> widget_width={widget_width}") # monitor
        if self._width > widget_width:
            self.h_scroll.pack(side=TOP, fill=X)
        else:
            self.h_scroll.pack_forget()
    def pack_vscrollbar(self):
        """Check if the vertical scrollbar is necessary and show/hide it accordingly"""
        print(f"self._length={self._length}") # monitor
        if self._length > 20:
            self.v_scroll.pack(side=RIGHT, fill=Y)
        else:
            self.v_scroll.pack_forget()
    def get_width(self):
        df = self._df
        return 600
        #return sum([max(df.loc[:,column].astype('str').str.len()) for column in df.columns]) # row length in pandas table
    def get_length(self):
        try:
            return min(1 + self._df.count('\n'),40)
        except:
            return min(1 + len(self._df),40)
    def set(self,text:str):
        def set_colors(text_widget):
            def set_lines_color(text_widget):
                lines = text_widget.get().splitlines()
                for i, line in enumerate(lines):
                    if i % 2 != 0:
                        text_widget.tb.tag_add("odd", f"{i+1}.0", f"{i+1}.end")
                        text_widget.tb.tag_config("odd",background=CONFIG['table']['alt_row_color'])
            def set_headlines_color(text_widget):
                text_widget.tb.tag_add("headline", f"1.0", f"1.end")
                text_widget.tb.tag_config("headline",foreground=CONFIG['table']['font_color'],background='white',font=(CONFIG['table']['font'],CONFIG['table']['font_size'],'bold'))
            def set_index_color(text_widget):
                rows = text_widget.get().splitlines()
                for i,row in enumerate(rows):
                    text_widget.tb.tag_add("index", f"{i+1}.0", f"{i+1}.{row.find(' ')}")
                    text_widget.tb.tag_config("index",foreground=CONFIG['table']['font_color'],font=(CONFIG['table']['font'],CONFIG['table']['font_size'],'bold'))
            
            try:
                set_lines_color(text_widget=self)
                set_headlines_color(text_widget=self)
                set_index_color(text_widget=self)
            except:
                pass 
        
        self.tb.insert("1.0",self._df)
        set_colors(text_widget=self)
        self.tb.configure(state='disabled')
class ChartOutput(Frame):
    def __init__(self, cmd, df:pd.DataFrame,x:str=None, master=None, *args, **kwargs):
        super().__init__(master, *args, **kwargs)

        # attributes
        self._df = df
        self._cmd = cmd
        self._x = x
        self._by = None

        # structure
        self.configure(bg='white')

        self.left_frame = Frame(self,width=15,bg='white')
        self.left_frame.pack(side=LEFT,padx=5,pady=2,expand=False)

        self.plot_frame = Frame(self)
        self.plot_frame.pack(side=LEFT,fill=X, expand=True, padx=1, pady=1)

        self.log = TextOutput(self.left_frame,width=30)
        self.log.pack(side=TOP,padx=2,pady=2)

        self.by_list = Listbox(
            self.left_frame,
            height=12,
            width=30,
            relief='flat',
            highlightcolor=CONFIG['code_block']['highlight_color'],
            font=(CONFIG['table']['font'],CONFIG['table']['font_size']-2),
            bd=1,
            bg=CONFIG['code_block']['background']
            )
        self.by_list.pack(side=TOP,padx=2,pady=2,fill=BOTH)

        # initiate 
        self.log.set(self._cmd)
        self.set_plot()
        self.set_by_list(self._df.columns.tolist())

        # key binds
        self.by_list.bind('<Double-1>',self.set_by_item)

    def set_log(self,text:str):
        self.log.set(text)    
    def set_plot(self):
                
        # Create a Seaborn plot
        #fig = get_dist_plot(df=self._df,x=self._x,by=self._by)
        fig = self._cmd # test
        # Create a canvas to embed the plot
        canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=BOTH, expand=True)
        
        # Close the figure to free memory
        plt.close(fig)    
    def set_by_list(self,items):
        MAX_UNIQUE = 20
        df = self._df
        for i,item in enumerate(items):
            data_type = str(df[item].dtype)
            if data_type in ['object','category','bool']:
                if len(df[item].unique()) <= MAX_UNIQUE:
                    self.by_list.insert(i,f"{item}")  
        self.by_list.insert(i+1,"None")           
    def set_by_item(self,event):
        selected_item = self.by_list.get(self.by_list.curselection()[0])
        self._by = selected_item
        self._log = f"X = '{self._x}'\nBy = '{self._by}'"
        self.log.set(self._log)
        self.plot_frame.pack_forget()
        self.plot_frame = Frame(self)
        self.set_plot()
        self.plot_frame.pack(side=RIGHT,fill=BOTH, expand=True, padx=1, pady=1)


# advance widgets
class CommandBlock(ctk.CTkFrame):
    def __init__(self, master=None, data=DATA_TABLE, cmd_string:str='', width:int=1000,id:int=0, **kwargs):
        super().__init__(master, **kwargs)
        
        self._id = id

        self.command_string = cmd_string
        self.data = data
        self.data_object = self.get_data_object()
        self.command = self.command_string.split('.')[-1]
        self._saved = False

        self.main_frame = ctk.CTkFrame(self,
                                       fg_color='white',
                                       #width=500,
                                       bg_color='white',
                                       border_color=get_darker_color(CONFIG['code_block']['background']),
                                       border_width=1,
                                       corner_radius=0
                                       )
        self.main_frame.pack(fill=X,expand=True,padx=1,pady=1)

        self.code_frame = ctk.CTkFrame(self.main_frame,
                                       width=width + 4*40,
                                       fg_color='transparent',
                                       bg_color='transparent',
                                       #border_width=0,
                                       corner_radius=0
                                       )
        self.code_frame.pack(side=TOP,padx=1,pady=1)

        self.entry = CodeLine(self.code_frame,
                               id=id,
                               width=width
                              )
        self.entry.set(cmd_string)
        self.entry.pack(side=LEFT,padx=1,pady=1)

        self.run = ctk.CTkButton(self.code_frame,
                                  text='>',
                                  width=30,
                                  border_width=1,
                                  border_color=get_darker_color(CONFIG['code_block']['run_color'],percentage=20),
                                  corner_radius=0,
                                  fg_color=CONFIG['code_block']['run_color'],
                                  hover_color=CONFIG['code_block']['background'],
                                  text_color=CONFIG['code_block']['font_color'],
                                  command=self.run_command
                                  )
        self.run.pack(side=LEFT,padx=1)  

        self.save = ctk.CTkButton(self.code_frame,
                                  text='+',
                                  width=30,
                                  border_width=1,
                                  border_color=get_darker_color(CONFIG['code_block']['save_color'],percentage=10),
                                  corner_radius=0,
                                  fg_color=CONFIG['code_block']['save_color'],
                                  hover_color=CONFIG['code_block']['background'],
                                  text_color=CONFIG['code_block']['font_color'],
                                  command=self.save_block
                                  )
        self.save.pack(side=LEFT,padx=1)      

        self.delete = ctk.CTkButton(self.code_frame,
                                  text='-',
                                  width=30,
                                  border_width=1,
                                  border_color=get_darker_color(CONFIG['code_block']['delete_color'],percentage=10),
                                  corner_radius=0,
                                  fg_color=CONFIG['code_block']['delete_color'],
                                  hover_color=CONFIG['code_block']['background'],
                                  text_color=CONFIG['code_block']['font_color'],
                                  command=self.delete_block
                                  )
        self.delete.pack(side=LEFT,padx=1)  

        self.duplicate = ctk.CTkButton(self.code_frame,
                                  text='++',
                                  width=30,
                                  border_width=1,
                                  border_color=get_darker_color(CONFIG['code_block']['duplicate_color'],percentage=10),
                                  corner_radius=0,
                                  fg_color=CONFIG['code_block']['duplicate_color'],
                                  hover_color=CONFIG['code_block']['background'],
                                  text_color=CONFIG['code_block']['font_color'],
                                  command=self.duplicate
                                  )
        self.duplicate.pack(side=LEFT,padx=1)                     
        
        self.output_frame = ctk.CTkFrame(self.main_frame,border_width=0,fg_color='white')
        self.output_frame.pack(side=TOP,fill=BOTH,expand=True,padx=1,pady=1)

        self.fold = ctk.CTkButton(self.output_frame,
                                  text='>',
                                  width=10,
                                  corner_radius=100,
                                  fg_color='white',
                                  hover_color=CONFIG['code_block']['background'],
                                  text_color='green',
                                  command=self.fold_output
                                  )
        self.fold.pack(side=LEFT,pady=1,padx=1)

        self.output = ctk.CTkTextbox(self.output_frame,
                                     wrap='none',
                                     height=20,
                                     fg_color='transparent',
                                     border_color='white',
                                     text_color=CONFIG['code_block']['font_color'],
                                     font=(CONFIG['code_block']['font'],CONFIG['code_block']['font_size']),
                                     bg_color='transparent',
                                     state='disable'
                                    )
        self.output.pack(side=LEFT,pady=1,padx=100,fill=BOTH,expand=True)

        self.comment = ctk.CTkTextbox(self.main_frame,
                                     wrap='word',
                                     width=width,
                                     text_color='#99A3A4',
                                     corner_radius=5,
                                     border_width=1,
                                     border_color=get_darker_color(CONFIG['code_block']['comment_color'],percentage=20),
                                     fg_color=CONFIG['code_block']['comment_color'],
                                     height=10,
                                     font=(CONFIG['code_block']['font'],CONFIG['code_block']['font_size']+1),
                                     bg_color='transparent',
                                     #state='disable'
                                    )
        self.comment.insert('0.0','<Comment>')                            
        self.comment.pack(side=TOP,padx=3,pady=4)

        #self.entry.bind('<Return>',lambda self.run_command)
        self.run_command()
        self.output_string = self.output.get('1.0',END)

    def save_block(self):
        #print('save block') # monitor
        if self._saved == False:
            self._saved = True
            self.main_frame.configure(border_width=2) #get_darker_color(CONFIG['code_block']['background'],percentage=80)) 
        else:
            self._saved = False

        #print(self._saved)
        self.main_frame.configure(border_width=1)        
    def duplicate(self):
        CommandBlock(self.master,cmd_string=f'>> {self.data["file_name"]}.{self.command}').pack(side=TOP,fill=X,expand=True,padx=5,pady=3)
    def delete_block(self):
        self.pack_forget()
    def get_data_object(self):
        if '[' in self.command_string.split('.')[0] and ']' in self.command_string.split('.')[0]:
            return 'column'
        else:
            return 'df'    
    def set_output(self,output_string):
        def get_height(output_string):
            LINE_HEIGHT = CONFIG['code_block']['font_size'] + 5
            rows = output_string.count('\n')
            return 3 + LINE_HEIGHT + rows*(LINE_HEIGHT-1)

        self.output.configure(state='normal')
        self.output.delete('0.0',END)
        self.output.insert('0.0',output_string)
        self.output.configure(height=get_height(output_string))
        self.output.configure(state='disable')
    def fold_output(self):
        if self.fold.cget('text') == '>':
            self.fold.configure(text='v')
            self.set_output(output_string='...')
        elif self.fold.cget('text') == 'v':
            self.fold.configure(text='>') 
            self.set_output(output_string=self.output_string)
    def run_command(self):

        self.command_string = self.entry.get()
        self.data_object = self.get_data_object()
        self.command = self.command_string.split('.')[-1]

        #print(f"Run: {self.command}") # monitor
        df = self.data['df']

        if self.data_object == 'df':
            if self.command == 'shape':
                self.set_output(f"{df.shape}")
            elif self.command == 'info()':
                self.set_output(f"{get_columns_info(df)}")
            elif self.command == 'columns':
                self.set_output(f"{df.columns.tolist()}")    
            elif self.command == 'describe()':
                self.set_output(f"{df.describe().T}")
            elif self.command == 'head()': 
                self.set_output(f"{df.head()}")      
            elif self.command == 'tail()': 
                self.set_output(f"{df.tail()}")       
            elif re.match(r'^head\(.*\)$',self.command) is not None: 
                AMOUNT = int(self.command[self.command.find('(')+1:self.command.find(')')])
                self.set_output(f"{df.head(AMOUNT)}") 
            elif 'loc[' in self.command and ']' in self.command:  
                try:
                    START_INDX = int(self.command[self.command.find('loc[') + len('loc[')])
                    END_INDEX = int(self.command[self.command.find(':')+1:self.command.find(',')])
                    self.set_output(f">> df.loc[{df.index[START_INDX]}:{df.index[END_INDEX]},]\n\n{df.loc[df.index[START_INDX]:df.index[END_INDEX],:]}")    
                except:
                    self.set_output(f">> Bad Index values: {START_INDX} - {END_INDEX}")   
            else:
                self.set_output(f"Unknown command: {self.command}")
        
        elif self.data_object == 'column':
            column = self.command_string[self.command_string.find('[')+1:self.command_string.find(']')]
            if self.command == 'describe()':
                out_data = df[column].describe().T.to_dict()
                self.set_output(f"{pd.DataFrame(index=out_data.keys(),data=out_data.values(),columns=[column])}")
            elif self.command == 'get_dist_plot()':
                output_parent = self.output.master
                #self.output.pack_forget()
                #self.output = get_dist_plot(output_parent,df=df,x=column) #need to fix
                self.set_output(f"chart")    
class CommandBlock(Frame):
    def __init__(self, master=None, data=DATA_TABLE, cmd_string:str='',id:int=0,x:str=None, commands=COMMANDS, **kwargs):
        super().__init__(master, **kwargs)     

        self._commands = commands
        self._id=id
        self._cmd_string = cmd_string
        self._df = data['df'] 
        self._x = x 
        self._data_object = 'df' if self._x == None else 'column'
        self._cmd = commands[self._data_object][cmd_string] if cmd_string in commands[self._data_object].keys() else cmd_string      

        self.configure(bg='white',
                       bd=1,
                       #highlightbackground = 'green',
                       highlightbackground= get_darker_color(CONFIG['code_block']['background'],percentage=10),
                       highlightthickness=1
                       )

        self.controls = CodeControls(self)
        self.controls.pack(side=TOP,expand=True,fill=X)

        self.code_line = CodeLine(self,id=self._id,width=150)
        self.code_line.set(cmd_string)
        self.code_line.pack(side=TOP,fill=X,expand=True) 

        self.out_frame = Frame(self,bg='white')
        self.out_frame.pack(side=TOP,fill=BOTH,expand=True)             

        self.output = Frame(self.out_frame)
        self.output.pack(side=LEFT)

        #self.run_command(event=None) # initiate code_block

        # binds
        self.code_line.entry.bind('<Return>',self.run_command)
        self.controls.run.bind('<Button-1>',self.run_command)
        self.controls.save.bind('<Button-1>',self.save_block)
        self.controls.delete.bind('<Button-1>',self.delete_block)
        self.controls.up.bind('<Button-1>',self.up)
        self.controls.down.bind('<Button-1>',self.down)

        # initiate code block
        #self.run_command(event=None)

    # controls
    def run_command(self,event):    
        def set_output(cmd_string,output_box=self.output):
            
            df = self._df
            x = self._x

            if '_plot' in cmd_string:
                #print(cmd_string) #
                try:
                    output_box = ChartOutput(master=output_box.master,df=df,x=x,cmd=exec(cmd_string))
                except Exception  as e:    
                    output_box = TextOutput(output_box.master,text=f'> Backend command to process:\n   {cmd_string}\n> Output error:\n   {e}')
            else:
                try:
                    output_box = TableOutput(output_box.master,df=eval(cmd_string))
                except Exception  as e:    
                    output_box = TextOutput(output_box.master,text=f'> Backend command to process:\n   {cmd_string}\n> Output error:\n   {e}')

            output_box.pack(side=TOP,padx=50,fill=X,expand=True)

        self.controls.progress_bar['value'] = 50
        # updating values per user input
        print(self.code_line.get_code()) # monitor
        self._cmd_string = self.code_line.get_code()
        self._cmd = self._commands[self._data_object][self._cmd_string] if self._cmd_string in self._commands[self._data_object].keys() else None

        # detecting code parts wth colors
        if '>>' in self.code_line.get():
            self.code_line.set(self.code_line.get())
        else:    
            self.code_line.set(f">> {self.code_line.get()}")

        try:    
            self.code_line.set_colors()
        except:
            pass    

        # processing command
        set_output(cmd_string=self._cmd,output_box=self.output)
        self.controls.progress_bar['value'] = 99

        return 'break'
    def save_block(self,event):
        code_block_list = self.master.winfo_children()
        self.configure(bd=2,
                       #highlightbackground=CONFIG['code_block']['highlight_color'],
                       highlightthickness=2
                       )

        # rearranged blocks
        for code_block in code_block_list:
            if code_block != self:
                code_block.pack_forget()

        self.pack(side=TOP,fill=X,expand=True,padx=5,pady=5)
        for code_block in code_block_list:
            if code_block != self:
                code_block.pack(side=TOP,fill=X,expand=True,padx=5,pady=5)
    def delete_block(self,event):
        self.destroy()
    def up(self,event):

        code_block_list =  sorted(self.master.winfo_children(), key=lambda w: (w.grid_info().get('row', 0), w.grid_info().get('column', 0))) # get current items by grid order (dont work with pack)
        i = code_block_list.index(self)

        if i > 0:
            code_block_list[i], code_block_list[i-1] = code_block_list[i-1], code_block_list[i]

            for j,code_block in enumerate(code_block_list):
                code_block.grid_forget()
                code_block.grid(row=j,column=0,padx=5,pady=5,sticky='nw')
    def down(self,event):

        code_block_list =  sorted(self.master.winfo_children(), key=lambda w: (w.grid_info().get('row', 0), w.grid_info().get('column', 0))) # get current items by grid order (dont work with pack)
        i = code_block_list.index(self)

        if i+1 < len(code_block_list):
            code_block_list[i], code_block_list[i+1] = code_block_list[i+1], code_block_list[i]

            for j,code_block in enumerate(code_block_list):
                code_block.grid_forget()
                code_block.grid(row=j,column=0,padx=5,pady=5,sticky='nw')

class FileExplorer(ctk.CTkFrame):
    def __init__(self, file_view_frame:ctk.CTkFrame, column_view_frame:ctk.CTkFrame, master=None, width:int=300, path:str='', **kwargs):
        super().__init__(master, **kwargs)
        
        self.file_view_frame = file_view_frame
        self.column_view_frame = column_view_frame

        self.entry = Entry(self,
                           width=width
                          )   
        self.entry.set(path)
        self.entry.pack(side=TOP,padx=1,pady=1)

        # Define style for Treeview
        style = ttk.Style()
        style.configure("Treeview", 
                        background=CONFIG['background'],
                        foreground=CONFIG['font_color'],
                        rowheight=22,
                        font=(CONFIG['font'],CONFIG['font_size']))
        style.map("Treeview", 
                  background=[('selected', get_darker_color(CONFIG['code_block']['background']))],
                  foreground=[('selected', 'black')]
                  )                

        self.treeview = ttk.Treeview(self,
                                    height=50,
                                    show='tree'
                                    )
        self.treeview.pack(side=TOP,fill=BOTH,expand=True,padx=1,pady=1)

        if path != '':
            self.set_content(self.entry.get())

        # key binds
        self.treeview.bind('<Double-1>',self.set_item)
        self.entry.bind('<Return>',self.update)

    def set_style(self):
        style = ttk.Style()
        style.configure("Treeview", 
                        background=CONFIG['background'],
                        foreground=CONFIG['font_color'],
                        rowheight=22,
                        font=(CONFIG['font'],CONFIG['font_size']))
        style.map("Treeview", 
                  background=[('selected', get_darker_color(CONFIG['code_block']['background']))],
                  foreground=[('selected', 'black')]
                  )  
    def update(self,event):
        path = self.entry.get()
        file_name = path.split('/')[-1]
        file_type = get_file_type(path)

        if file_name == '...':
            parent_dir_path = '/'.join(path.split('/')[:-2]) + '/'
            self.entry.set(parent_dir_path)
            self.set_content(path=parent_dir_path)
            return

        if file_type == 'dir':
            self.set_content(path=path)
            return

        if file_type in CONFIG['supported_files']: 
            print(f"Loading {DATA_TABLE['path']}")
            DATA_TABLE['path'] = path
            DATA_TABLE['file_name'] = file_name
            DATA_TABLE['df'] = read_data_file(DATA_TABLE['path'])

            self.file_view_frame.pack_forget() 
            self.file_view_frame = FileView(self.file_view_frame.master,data=DATA_TABLE)
            self.file_view_frame.set(cmd_list=COMMANDS['df'].keys())
            self.file_view_frame.pack(expand=True,fill=BOTH)

            self.column_view_frame.pack_forget()
            self.column_view_frame = ColumnView(self.column_view_frame.master,data=DATA_TABLE)
            self.column_view_frame.set()
            self.column_view_frame.pack(expand=True,fill=BOTH)

            self.set_style() # test

    def set_content(self,path:str):
        self.treeview.delete(*self.treeview.get_children())

        for file in ['...'] + get_dir(path):
            if get_file_type(file) in CONFIG['supported_files']:
                self.treeview.insert('','end',text=file,tags=('data_file',))
            else:
                self.treeview.insert('','end',text=file)
        self.treeview.tag_configure("data_file", foreground="green")

        self.set_style() # test
    def get_path(self):
        return self.entry.get()
    def get_item(self):
        return self.treeview.selection()[0]
    def set_item(self, event):
        #print(">> pressed double click on file in list") # monitor
        entry_string = self.entry.get()
        current_path = '/'.join(self.entry.get().split('/')[:-1]) + '/'
        file_name = self.treeview.item(self.treeview.selection(),'text')

        self.entry.set(f"{current_path}{file_name}")
        self.update(event)
        #print(f">> type={file_type}, go_back={file_path.endswith('...')}") # monitor
class FileView(ctk.CTkFrame):
    def __init__(self, master=None, data = DATA_TABLE, **kwargs):
        super().__init__(master, **kwargs)

        self.df = data['df']
        self.filename = data['file_name'].split('.')[0] if data['file_name']!=None else None

        self.frame = ctk.CTkScrollableFrame(self,
                                            label_text=f'File Overview (df = {self.filename})' if self.filename!=None else 'Pick Data File',
                                            label_fg_color = 'transparent',
                                            label_text_color=CONFIG['code_block']['font_color'],
                                            label_font=(CONFIG['code_block']['font'],CONFIG['code_block']['font_size']+2),
                                            height=940,
                                            corner_radius=5,
                                            border_width=0,
                                            #border_color='green',
                                            #scrollbar_fg_color='white',
                                            #scrollbar_button_hover_color='red',
                                            #scrollbar_button_color='blue',
                                            fg_color=CONFIG['background']
                                            )
        self.frame.pack(fill=BOTH,expand=True,padx=3,pady=3)

    def set(self, cmd_list:list):
        for i,cmd in enumerate(cmd_list):
            CommandBlock(self.frame,id=i,cmd_string=cmd).grid(row=i,padx=5,pady=5,sticky='nw')
        
class ColumnView(ctk.CTkFrame):
    def __init__(self, master=None, data = DATA_TABLE, commands=COMMANDS, **kwargs):
        super().__init__(master, **kwargs)

        self._commands = commands
        self._df = data
        self._file_name = data['file_name'].split('.')[0] if data['file_name'] != None else None
        self._column = None
        WIDTH = (10 + max([len(col) for col in data['df'].columns]))*8 if len(data['df'].columns) > 0 else 250 # get max width for control panel

        self.control_frame = ctk.CTkFrame(self,width=WIDTH)
        self.control_frame.pack(side=LEFT)

        self.entry = Entry(self.control_frame,width=WIDTH*(1.2))
        self.entry.pack(side=TOP,fill=X,expand=True,padx=1,pady=1)

        # Define style for Treeview
        style = ttk.Style()
        style.configure("Treeview", 
                        background=CONFIG['background'],
                        foreground=CONFIG['font_color'],
                        rowheight=25,
                        font=(CONFIG['font'],CONFIG['font_size']-2))
        style.map("Treeview", 
                  background=[('selected', 'blue')],
                  foreground=[('selected', 'white')]
                  )                
        
        self.treeview = ttk.Treeview(self.control_frame,
                                    height=50,
                                    show='tree'
                                    )
        self.treeview.pack(side=TOP,fill=BOTH,expand=True,padx=1,pady=1)

        self.frame = ctk.CTkScrollableFrame(self,
                                            label_text="Column's Overview",
                                            label_fg_color = 'transparent',
                                            label_text_color=CONFIG['code_block']['font_color'],
                                            label_font=(CONFIG['code_block']['font'],CONFIG['code_block']['font_size']+2),
                                            height=940,
                                            corner_radius=5,
                                            border_width=0,
                                            #border_color='green',
                                            #scrollbar_fg_color='white',
                                            #scrollbar_button_hover_color='red',
                                            #scrollbar_button_color='blue',
                                            fg_color=CONFIG['background']
                                            )
        self.frame.pack(side=LEFT,fill=BOTH,expand=True,padx=3,pady=3)

        self.treeview.bind('<Double-1>',self.get_column) # Bind the double click event to the listbox

    def get_column(self, event):
        selected_item = self.treeview.item(self.treeview.selection(),'text')
        self._column = selected_item[selected_item.find(')')+2:]
        self.entry.set(self._column)
        #print(f"set new cmd: file={self._file_name}, column={self._column}") # monitor
        # need to change all X in command blocks to column name  ###########################################
        try:
            self.frame.pack_forget()
        except:
            pass    
        self.set_column()  

    def set_column(self):
        self.frame.configure(label_text=f"Column Overview(df = {self._file_name}, X ='{self._column}')")

        self.frame = ctk.CTkScrollableFrame(self,
                                            label_text="Column's Overview",
                                            label_fg_color = 'transparent',
                                            label_text_color=CONFIG['code_block']['font_color'],
                                            label_font=(CONFIG['code_block']['font'],CONFIG['code_block']['font_size']+2),
                                            height=940,
                                            corner_radius=5,
                                            border_width=0,
                                            #border_color='green',
                                            #scrollbar_fg_color='white',
                                            #scrollbar_button_hover_color='red',
                                            #scrollbar_button_color='blue',
                                            fg_color=CONFIG['background']
                                            )
        self.frame.pack(side=LEFT,fill=BOTH,expand=True,padx=3,pady=3)

        i=0
        for cmd_key in self._commands['column'].keys():
            df = self._df
            column = self._column
            #print(f"id={i}, cmd={self._commands['column'][cmd_key]}, df={df}, column={column}") # monitor
            CommandBlock(self.frame,id=i,cmd_string=cmd_key,x=column).pack(side=TOP,fill=X,expand=True,padx=5,pady=5)
            i += 1

    def set(self):
        df = self._df['df']
        for column in df.columns:
            data_type = str(df[column].dtype)
            if data_type in CONFIG['data_types']:
                self.treeview.insert('','end',text=f"({data_type}) {column}",tags=(data_type,))
            else:
                self.treeview.insert('','end',text=column)
        
        for item in CONFIG['data_types'].keys():         
            self.treeview.tag_configure(item, foreground=CONFIG['data_types'][item])
           