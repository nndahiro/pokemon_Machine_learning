# pokemon_Machine_learning
A machine learning approach to predicting pokemon types based on a database of pictures.

***Background***

These days there are almost too many pokemon to account with almost 900 different kind
span over 8 generations (and counting). Yet, with so many choices to pick from, when playing
the game it is of utmost importance to have a balanced team. Pokemon are divided into 18
different **types** broadly based off "elemental" or conceptual ideas. These include: fire, grass,
water, fighting, fairy, flying and more...
The challenge now becomes, how to tell a pokemon type simply by looking at it? Is it even possible?

These types can usually be discriminated from by the human eye: Fire types have flames,
water types are blue...
However, as more and more pokemon are created by Nintendo and GameFreak, it is less and less easy
to simply look at a pokemon and predict the type. As a fan of the series I prided myself on
being able to predict a pokemon's type by looking at it. So I wondered if it was possible to teach a
computer to learn the same.

As it is the 21st century and humanity is enjoying a period of "Throw AI at it" I decided to build a Neural
Network to 1) Learn how to do this, and 2) see if neural nets can solve this problem.

The data I acquired to tackle this problem is from Kaggle.com and consists of 100x100 and 120x120 sized images.
Recently, neural networks have been getting better at classifying images, specifically, convolutional neural
networks(CNNs). They are good at processing high dimensional data cmpared to using Dense/classical neural networks.
So I decided to use this structure.
Pokemon can be up to 2 **types**(by this I mean the pokemon's "element") but can also be 1 type or a pure type.
This means that each pokemon image, from the standpoint of a neural network model, is a multi-class sample.
So now, our problem is shaping up to be a **Multi-class, multi-label, image classification** problem with 18 classes.
(There are 18 Pokemon Types).

I created different models (some better than others) and I also build a GUI to make it easier for people
to use the model I generated (models are in the Models folder).
You can Download your own pictures and use the GUI to predict a Pokemon's type.


***Requirements***

TensorFlow instructions can be found at https://www.tensorflow.org/install/pip?lang=python3 whether you have Windows, Linux or Mac
  Note that TensorFlow is not yet compatible with Python 3.8 (As of 6th May 2020) but works with 3.74.
  
Keras module: " pip install keras " or go to https://keras.io/#installation

PIL module: " pip install Pillow"

Jupyter Notebook Software: if you want to visualize the data manipulation from databases.

Other standard python modules such as scikit, tkinter

***Running***

Run GUI.py


Open the GUI.py scrypt and change the model it runs on to see the effect.

The "Pokemon_database_pandas_manipulation" Jupyter Notebook show the extraction of images and labels from the databases.

The "Model Testing Notebook" shows the testing of the model as well as

pokemonIt would be nice indeed to be able to predict a pokemon type just from a picture

![](In%20class%20Testing/charizard.jpg)

Multiple independent classif and gives probability. Sigmoid gives independent
