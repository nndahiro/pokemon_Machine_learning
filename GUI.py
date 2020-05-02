import matplotlib.pyplot as plt
import turtle
import os
from tkinter import *
from tkinter import filedialog
from PIL import ImageTk, Image
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from Pokemon_ML_build import pokemon_type_predict as ptp
import numpy as np
from Image_processing import array_image, image_array_resize


if __name__ == '__main__':
    class TurtleGui():
        def __init__(self, turtle):
            self.t = turtle
        def forwardi(self):
            self.t.forward(distance=20)
        def turn_left(self):
            self.t.left(30)
        def turn_right(self):
            turtle.right(30)
        def translate(x,y):
            turtle.goto(x,y)
        def sleep(self):
            turtle.bye()
    # turtle.setup(300,300)
    # window = turtle.Screen()
    # window.title('GUI first ting')
    # window.bgcolor("blue")
    # tortue_genial = TurtleGui(turtle.Turtle())
    # tortue_genial.t.color('green')
    # tortue_genial.t.shape('turtle')
    # tortue_genial.t.pensize(5)
    # window.onkey(tortue_genial.forwardi, "Up")
    # window.onkey(tortue_genial.turn_left, "Left")
    # window.onkey(tortue_genial.sleep, "s")
    # window.onclick(tortue_genial.translate) #guess it takes coordinates as input
    # window.listen()
    # window.mainloop()
    #dark colors: dragon, dark, ghost, 
    dict_color = {'bug': "#666600",
                    'dark': "black",
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
        def __init__(self, buttons=[], panels=[]):
            ''' Will define main GUI Characterisitics like
            Size, icon, background, name, shape/columns?
            '''
            super(GUI, self).__init__() # All inherited functionality of Tk class
            self.title("Pokemon: \n AI GUI (TM)")
            self.minsize(200,200)
            self.buttons = buttons
            self.panels = panels

        def button(self, name, b_row, b_column, type_of_button, entry_box=None):
            ''' Create a button, specify name, location and type
            '''
            if type_of_button.lower() == 'browse':
                button = Button(self, text= name, padx=30, pady=10,
                             command = lambda: self.file_explorer(entry_box)) #Can't call function so Have to use lambda
                self.buttons.append([name,b_row,b_column, type_of_button])
            elif type_of_button.lower() == "predict":
                type_of_b = None #TODO
                self.loadimage = PhotoImage(file="C:/Users/Nelson/Pictures/pokeball2.png")
                button = Button(self, image=self.loadimage,
                             command = lambda: self.image_plotter_predictor(entry_box))
                button["border"] = "0" #No borders
                self.buttons.append([name,b_row,b_column, type_of_button])
                # self.roundedbutton["bg"] = "white"
                # self.roundedbutton["border"] = "0"
                # self.roundedbutton.pack(side="top")
                # button = Button(self, text= name, padx=40, pady=20)
            elif type_of_button.lower() == "try again":
                ''' Remove all panels and start over'''
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
            if len(entry_box.get()) > 0:
                try:
                    image_name = str(entry_box.get())
                    #Use PIL module so that this works with all image formats:
                    img = ImageTk.PhotoImage(Image.open(image_name).resize((200,200)))
                    panel = Label(self, image=img)
                    panel.image = img 
                    # ^THIS IS needed as a reference python will get rid of image while parsing: Tkinter issue
                    #ref http://effbot.org/pyfaq/why-do-my-tkinter-images-not-appear.htm
                    panel.grid(row = 4, column = 0, columnspan= 3)
                    #Now that images is loaded, we also put a message:
                    msg = Label(self, text= "Your Picture:", bg="#7ABBEC")
                    msg.grid(row=3, column= 1, columnspan=1)
                    print("Picture In")
                    predict_img = array_image(image_name) #turn to array so model can use it
                    predict_img_resized_for_model = image_array_resize(predict_img, (100,100,3))
                    #predict_img_resized_for_model = np.expand_dims(predict_img_resized_for_model, axis=0)
                    # ^Reshaped to (1,100,100,3) for model
                    type1, type2 = ptp(predict_img_resized_for_model)
                    type1_panel = Label()
                    type1_panel.grid(row=5, column=0)
                    type2_panel = Label()
                    type2_panel.grid(row=5, column=0)
                    #One way of removing labels is to set it to be blank
                    #Then deleting it everytime before reassigning a new
                    #label
                    if type2 == None:
                        # global type1_panel
                        # global type2_panel
                        type1_panel.destroy()
                        type2_panel.destroy()
                        message = Label(text="Your prediction:", padx = 50, pady=30)
                        message.grid(row=5, column=0,columnspan=1)
                        type1_panel = Label(text=type1, bg=dict_color[type1], padx=40, pady=20, font=0.6)
                        type1_panel.grid(row=5, column=1, columnspan=1)
                    else:
                        # global type1_panel
                        # global type2_panel
                        type1_panel.destroy()
                        type2_panel.destroy()
                        message = Label(text="Your prediction:", padx = 50, pady=30)
                        message.grid(row=5, column=0,columnspan=1)
                        type1_panel = Label(text=type1, bg=dict_color[type1], padx=40, pady=20, font=0.6)
                        type1_panel.grid(row=5, column=1, columnspan=1)
                        type2_panel = Label(text=type2, bg=dict_color[type2], padx=40, pady=20, font=0.6)
                        type2_panel.grid(row=5, column=2, columnspan=1)
                    


                    # print(predict_img)

                except FileNotFoundError: #IF file not found:
                    panel = Label(self, text= "No such Directory, try again")
                    panel.grid(row= 4, column =0, columnspan= 3)
                    print("That file path does not exist")
                    return
            else:
                return #If nothing is entered, do nothing

        def panel(self):
            return            




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