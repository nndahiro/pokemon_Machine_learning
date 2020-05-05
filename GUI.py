import matplotlib.pyplot as plt
import turtle
import os
from tkinter import *
from tkinter import filedialog
from PIL import ImageTk, Image
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from Pokemon_ML_build import pokemon_type_predict as ptp
import numpy as np
from Image_processing import array_image, image_array_resize, centralize_image


if __name__ == '__main__':
    #dark colors: dragon, dark, ghost, 
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
        def __init__(self, buttons=[], panels=[], images=[]):
            ''' The GUI class is a class that inherits all attributes
            of the Tk class from tkinter module with additional
            functionality that we use. It is a Graphical User Interface
            with buttons, panels, images and an entrybox. These are all
            specified as the GUI class instance is built
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

        def button(self, name, b_row, b_column, type_of_button, entry_box=None):
            ''' Create a button, specify name, location and type
            '''
            if type_of_button.lower() == 'browse':
                button = Button(self, text= name, padx=30, pady=10,
                             command = lambda: self.file_explorer(entry_box)) #Can't call function so Have to use lambda
                self.buttons.append([button,name,b_row,b_column, type_of_button])
            elif type_of_button.lower() == "predict":
                type_of_b = None #TODO
                self.loadimage = PhotoImage(file=os.path.join(os.getcwd(),"pokeball2.png"))
                button = Button(self, image=self.loadimage,
                             command = lambda: self.image_plotter_predictor(entry_box))
                button["border"] = "0" #No borders
                self.buttons.append([button,name,b_row,b_column, type_of_button])
                # self.roundedbutton["bg"] = "white"
                # self.roundedbutton["border"] = "0"
                # self.roundedbutton.pack(side="top")
                # button = Button(self, text= name, padx=40, pady=20)
            elif type_of_button.lower() == "try again":
                ''' Button where command is to
                Remove all panels and start over'''
                for each_panel in self.panels:
                    each_panel.destroy()
            
            button.grid(row=b_row, column=b_column)
        def entry_box(self, e_row, e_column, span):
            entry = Entry(self, width=60)
            entry.grid(row=e_row, column=e_column, columnspan= span)
            return entry # So that you can access a specific entry object in GUI

        def file_explorer(self, entry_box, window_title="Choose Pokemon Picture"):
            filename = filedialog.askopenfilename(title=window_title, initialdir="/",
                                                     filetypes = (("jpeg files", "*.jpg"),("png files", "*.png"), ("numpy files", "*.npy")))
            # ^ Returns a string of path
            entry_box.delete(0, END)
            entry_box.insert(0, filename) #Need to specify which entry_box I am referring to
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
                    #Use PIL module so that this works with all image formats:
                    img = ImageTk.PhotoImage(Image.open(image_name).resize((200,200)))
                    if len(self.images) >0: #if there's already an image
                        self.images[0].destroy() #remove image
                        self.images.pop()
                        panel = self.panel(image=img) #put new one
                        panel.image = img #keep a ref
                    else: #no image prior
                        panel = self.panel(image=img)
                        panel.image = img 
                    # ^THIS IS needed as a reference python will get rid of image while parsing: Tkinter issue
                    #ref http://effbot.org/pyfaq/why-do-my-tkinter-images-not-appear.htm
                    panel.grid(row = 4, column = 0, columnspan= 3)
                    #Now that images is loaded, we also put a message:
                    msg = self.panel(text= "Your Picture:", bg="#7ABBEC")
                    msg.grid(row=3, column= 1, columnspan=1)
                    print("Picture In")
                    predict_img = array_image(image_name, verbose=False) #turn to array so model can use it
                    #Crop picture
                    #clean_image = centralize_image(predict_img)
                    predict_img_resized_for_model = image_array_resize(predict_img, (64,64,3))
                    #predict_img_resized_for_model = np.expand_dims(predict_img_resized_for_model, axis=0)
                    #TODO: Change to HSV if needed
                    type1, type2 = ptp(predict_img_resized_for_model)
                    # type1_panel = Label()
                    # type1_panel.grid(row=5, column=0)
                    # type2_panel = Label()
                    # type2_panel.grid(row=5, column=0)
                    #One way of removing labels is to set it to be blank
                    #Then deleting it everytime before reassigning a new
                    #label
                    if len(self.panels) >=3:#more than msg, type, "your predict": msg
                        print(f'Destroying panels... list= {len(self.panels)}')
                        for each_panel_besides_msg in range(len(self.panels)-1):
                            self.panels[1+each_panel_besides_msg].destroy()#Destroy except msg
                        # self.panels[2].destroy()
                        # self.panels[3].destroy()
                        # self.panels[4].destroy()
                        del self.panels[1:]#remove all panels from list except image and "your pic"
                        print(f'Destroy-ED panels... list= {len(self.panels)}')
                    if type2 == None:
                        # global type1_panel
                        # global type2_panel
                        message = self.panel(text="Your prediction:", padx = 30, pady=30, bg="#7ABBEC")
                        message.grid(row=5, column=0,columnspan=1)
                        type1_panel = self.panel(text=type1, bg=dict_color[type1], padx=40, pady=20)#, font="Helvetica")
                        type1_panel.grid(row=5, column=1, columnspan=1)
                    else:
                        print(len(self.panels))
                        message = self.panel(text="Your prediction:", padx = 50, pady=30, bg="#7ABBEC")
                        message.grid(row=5, column=0,columnspan=1)
                        type1_panel = self.panel(text=type1, bg=dict_color[type1], padx=40, pady=20)#, font="Helvetica")
                        type1_panel.grid(row=5, column=1, columnspan=1)
                        type2_panel = self.panel(text=type2, bg=dict_color[type2], padx=40, pady=20)#, font=0.6)
                        type2_panel.grid(row=5, column=2, columnspan=1)
                        print(len(self.panels))
                    


                    # print(predict_img)

                except FileNotFoundError: #IF file not found:
                    panel = Label(self, text= "No such Directory, try again")
                    panel.grid(row= 4, column =0, columnspan= 3)
                    print("That file path does not exist")
                    return
            else:
                return #If nothing is entered, do nothing

        def panel(self, text="", bg=None, padx=1, pady=1, image=None):#, font="Helvetica"):
            created_panel = Label(text=text, bg=bg, padx=padx, pady=pady, image=image)
            if image != None: #if its an image don't add to panels
                self.images.append(created_panel)
                return created_panel
            self.panels.append(created_panel)
            return created_panel     




    # root = Tk()
    # root.title('Tk')
    # entry =  Entry(root, width=50)
    # entry.grid(row=0, column=1, columnspan=2, padx=10, pady=10)
    
    # def function():
    #     print('Your Pokemon is type Fire, Flying')
    # button = Button(root, text= "Predict Pokemon Type!", padx= 40, pady=20, command=function)
    # button.grid(row=1, column=0)
    
    root = GUI()
    root.configure(bg="#7ABBEC")
    
    entry_b = root.entry_box(0,0,2)
    root.button('Browse', 0,2, type_of_button='browse', entry_box= entry_b)
    root.button("PokeAI", 1,1, type_of_button='predict', entry_box= entry_b)
    type1_panel = Label(root)
    type2_panel = Label(root) 
    #Don't do anything to it, will manipulate it in methods
    # We will destroy it in the method but we need it to be a global variable
    #So that program remembers it
    # typeo = "flying"
    # panelo = Label(text=typeo, bg=dict_color[typeo], padx=40, pady=20, font=0.6)
    # panelo.grid(row=4, column=1, columnspan=2)
    root.mainloop()