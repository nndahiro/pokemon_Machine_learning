'''
The GUI to be run is defined here and is ready
to be run.
'''
import matplotlib.pyplot as plt
import os
from tkinter import *
from tkinter import filedialog
from PIL import ImageTk, Image
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from keras.models import load_model
from Pokemon_ML_build import pokemon_type_predict as ptp
import numpy as np
from Image_processing import array_image, image_array_resize, centralize_image

dict_color = {'bug': "#666600",
                    'dark': "gray",
                    'dragon': "#000066",
                    'electric': "yellow",
                    'fairy': "lightpink",
                    'fighting': "#990000",
                    'fire': "#FF8000",
                    'flying': "#99CCFF",
                    'ghost': "#003366",
                    'grass': "green",
                    'ground': "#CC6600",
                    'ice': "#00FFFF",
                    'normal': "#E0E0E0",
                    'poison': "#990099",
                    'psychic': "pink",
                    'rock': "#994C00",
                    'steel': "#606060",
                    'water':"#0080FF",
                    'None': "#E0E0E0"}

class GUI(Tk):
    def __init__(self, poke_model, buttons=[], panels=[], images=[],):
        ''' The GUI class is a class that inherits all attributes
        of the Tk class from tkinter module with additional
        functionality that we use. It is a Graphical User Interface
        with buttons, panels, images and an entrybox. It also holds
        the particular predictive model that will run type prediction.
        These are all specified as the GUI class instance is built.
        **Parameters**
            buttons: *Tkinter Button Object*
                    This is a button that will be added to our
                    GUI through the .button method. Buttons can
                    call a command when they are clicked.
            panels: *Tkinter Panel Object*
                    This is text that appears in the GUI that is not
                    a button.
            images: *ImageTk PhotoImage Object*
                    This is an image that can be displayed in the GUI.
                    It can only be accessed with the PIL module.
        **Returns**
            None
            Simply instanciates the class object.
        '''
        super(GUI, self).__init__()
        self.title("Pokemon: AI GUI (TM)")
        self.minsize(200,500)
        self.buttons = buttons
        self.panels = panels
        self.images = images
        self.poke_model = poke_model

    def button(self, name, b_row, b_column, type_of_button, entry_box=None):
        ''' Create a button, specify name, location and type. Each type of
            button corresponds to a command which runs a function.
            **Parameters**:
                name: *str*
                    This is the name of button and text that will display.
                b_row: *int*
                    The row in the GUI window grid that the button will
                    be placed.
                b_column: *int*
                    The column in the GUI window grid that the button will
                    be placed.
                type_of_button: *str*
                    A string that will decide the command of the button you
                    create. Not case sensitive.
            **Returns**
                None
        '''
        if type_of_button.lower() == 'browse':
            button = Button(self, text= name, padx=30, pady=10,
                            command = lambda: self.file_explorer(entry_box))
                            #Can't call function so Have to use lambda
            self.buttons.append([button,name,b_row,b_column, type_of_button])
        elif type_of_button.lower() == "predict":
            self.loadimage = PhotoImage(file=os.path.join(os.getcwd(),"pokeball2.png"))
            button = Button(self, image=self.loadimage,
                            command = lambda: self.image_plotter_predictor(entry_box))
            button["border"] = "0" #No borders
            self.buttons.append([button,name,b_row,b_column, type_of_button])
            
        elif type_of_button.lower() == "random":
            button = Button(self, text=name, padx=20, pady=10)
            ''' Button where command is to
            Remove all panels and start over'''
            for each_panel in self.panels:
                each_panel.destroy()
        
        button.grid(row=b_row, column=b_column)
    def entry_box(self, e_row, e_column, span):
        entry = Entry(self, width=60)
        entry.grid(row=e_row, column=e_column, columnspan= span)
        return entry 
        # returns entry so that you can access
        # a specific entry object in GUI

    def file_explorer(self, entry_box, window_title="Choose a Pokemon Picture"):
        filename = filedialog.askopenfilename(title=window_title, initialdir="/",
                                                    filetypes = (("jpg files", "*.jpg"),
                                                    ("png files", "*.png"), 
                                                    ("numpy files", "*.npy"),
                                                    ("jpeg files", "*.jpeg")))
        # ^ Returns a string of path
        entry_box.delete(0, END) #removes what was previously there
        entry_box.insert(0, filename)
        return filename #So we have access to the path string

    def image_plotter_predictor(self, entry_box):
        '''
        This is the command of the 'predict' button of the GUI.
        It takes the path given by the entry box, retrieves the
        image and plots it as a panel in the GUI. It then converts
        the image to an array and changes its' shape to the model
        input shape. Then the array is run through the Neural network
        and the top1 or 2 pokemon types are displayed based on a thr
        -eshold. This button can be re-pressed within one mainloop.
        **Parameters**
            entry_box: *Tkinter Entry Object"
                    This is the entry where the image path is taken.
        **Returns**
            None

        '''
        if len(entry_box.get()) > 0:
            try:
                image_name = str(entry_box.get())
                #Use PIL module to create an image for the Tkinter module
                img = ImageTk.PhotoImage(Image.open(image_name).resize((200,200)))
                if len(self.images) >0: #if there's already an image
                    self.images[0].destroy() #remove image
                    self.images.pop()
                    panel = self.panel(image=img) #put new one
                    panel.image = img #keep a ref
                else: #no image prior
                    panel = self.panel(image=img)
                    panel.image = img 
                # ^THIS is needed as a reference because python will get rid
                # of image while parsing: Tkinter issue
                #ref http://effbot.org/pyfaq/why-do-my-tkinter-images-not-appear.htm
                panel.grid(row = 4, column = 0, columnspan= 3)
                #Now that images is loaded, we also put a message:
                msg = self.panel(text= "Your Picture:", bg="#7ABBEC")
                msg.grid(row=3, column= 1, columnspan=1)
                print("Picture In")
                # turn to array so model can use it
                predict_img = array_image(image_name, verbose=False)
                # Crop picture and augment/blur it.
                clean_image = centralize_image(predict_img)
                predict_img_resized_for_model = image_array_resize(clean_image, (64,64,3))
                try:
                    type1, type2 = ptp(predict_img_resized_for_model,trained_model=self.poke_model)
                except ValueError: #If you are running the 100x100 model
                    predict_img_resized_for_model = image_array_resize(clean_image, (100,100,3))
                    type1, type2 = ptp(predict_img_resized_for_model,trained_model=self.poke_model)
                
                if len(self.panels) >=3:
                    # Check if "Your prediction", "type1" and "your picture"
                    #  is already in the GUI display. If yes, destroy all
                    # except "Your picture"

                    for each_panel_besides_msg in range(len(self.panels)-1):
                        #Destroy everything except "Your picture" message
                        self.panels[1+each_panel_besides_msg].destroy()

                    del self.panels[1:]
                    # remove all panels from list except image and "your pic"
                if type2 == None:
                    message = self.panel(text="Your prediction:", padx = 30, pady=30, bg="#7ABBEC")
                    message.grid(row=5, column=0,columnspan=1)
                    type1_panel = self.panel(text=type1, bg=dict_color[type1], padx=40, pady=20)
                    type1_panel.grid(row=5, column=1, columnspan=1)

                else:

                    message = self.panel(text="Your prediction:", padx = 50, pady=30, bg="#7ABBEC")
                    message.grid(row=5, column=0,columnspan=1)
                    type1_panel = self.panel(text=type1, bg=dict_color[type1], padx=40, pady=20)
                    type1_panel.grid(row=5, column=1, columnspan=1)
                    type2_panel = self.panel(text=type2, bg=dict_color[type2], padx=40, pady=20)
                    type2_panel.grid(row=5, column=2, columnspan=1)
                

            except FileNotFoundError: #IF file not found:
                panel = Label(self, text= "No such Directory, try again")
                panel.grid(row= 4, column =0, columnspan= 3)
                print("That file path does not exist")
                return
        else:
            return #If nothing is entered, do nothing

    def panel(self, text="", bg=None, padx=1, pady=1, image=None):
        created_panel = Label(text=text, bg=bg, padx=padx, pady=pady, image=image)
        if image != None: 
            #if its an image don't add to panels, add to images
            self.images.append(created_panel)
            return created_panel
        self.panels.append(created_panel)
        return created_panel  



if __name__ == '__main__':
 
    cwd = os.getcwd()
    model_name = "Pokemon_ML_image_center.h5" #Change model to use here
    folder_path = os.path.join("Models",model_name)
    full_m_path = os.path.join(cwd, folder_path)
    p_model = load_model(full_m_path)

    
    root = GUI(poke_model=p_model)
    root.minsize(200,200)
    root.configure(bg="#7ABBEC")
    
    entry_b = root.entry_box(0,0,2) # place entry box
    root.button('Browse', 0,2, type_of_button='browse', entry_box= entry_b)  # place button
    root.button("PokeAI", 2,1, type_of_button='predict', entry_box= entry_b)
    welcome_label = Label(root, 
                        text="Hello Trainer! \n Please upload a picture by Browsing in your files \n and press the button below to predict pokemon type(s).",
                        bg="#7ABBEC", font= ("Helvetica", 11))
    welcome_label.grid(row=1,column=0, columnspan=3) # place welcome panel

    root.mainloop() # Run GUI loop