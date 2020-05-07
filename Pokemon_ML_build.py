import numpy as np
import matplotlib.pyplot as plt
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' #Suppress TF warning
import pickle
from sklearn.model_selection import train_test_split
from keras.metrics import categorical_accuracy
from keras.utils import np_utils
from keras.models import Sequential, load_model, Model
from keras.layers.core import Dense, Flatten, Dropout, Activation
from keras.layers import Conv2D, MaxPool2D, BatchNormalization, advanced_activations
import skimage.transform as transform
from copy import deepcopy
from Image_processing import image_array_resize, centralize_image, rgb_hsv

def get_key(val,my_dict):
    '''Get key from dictionary value
    '''
    for key, value in my_dict.items(): 
         if val == value: 
            return key

def load_obj(name ):
    with open('obj/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)
def one_hot_to_type(one_h, as_string = True):
    ''' The Y_data returns a label (pokemon type)
    in the form of a list so this is a way to go
    to the pokemon type using the 2 dictionary
    it returns a string of a list containing the
    types. Refer to
    "Pokemon_database_pandas_manipulation.ipynb"
    for more information on why we use 2
    dictionaries.
    **Parameters**:
        one_h: *np.array*,*int*
            The one-hot vector encoding the label
            of a sample. It has 18 numbers, a 1
            corresponds to a certain pokemon type
            being present in the picture. 0 means
            the type is not present. All
            pokemon images have at most 2 types.
        as_string: *bool*
            Whether to return a list or a string
            with the type information. Typically
            you only need the string
    **Returns**:
        type_string: *str*
            The decoded types in a string
        type_list: *list*,*str*
            The decoded types in a list
    '''
    indexer = get_key(one_h, index_to_one_hot)
    if as_string:
        type_string = f"{index_type_dict.get(indexer)}"
        return type_string
    type_list = index_type_dict.get(indexer)
    return type_list

def plot_performance(model, history, output_name):
    '''
    Retrieve accuracies from the history object and save them to a figure.

    **Parameters**

        model:
            A class object that holds the trained model.
        history:
            A class object that holds the history of the model.
        output_name: *str, optional*
            The filename of the output image.

    **Returns**

        None
    From Divya and Isaiah in Software Carpentry Class JHU, 2020.
    '''
    # Plot accuracy
    plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['train', 'test'], loc='lower right')
    # Plot loss
    plt.subplot(2, 1, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['train', 'test'], loc='upper right')
    plt.tight_layout()
    # Calculate loss and accuracy on the testing data
    loss, accuracy = model.evaluate(X_test, Y_test, verbose=2)
    print("Test Loss:", loss)
    print("Test Accuracy:", accuracy)
    # Save figure as .png file
    if output_name[-4:] != ".png":
        output_name += ".png"
    fig = plt.gcf()
    fig.savefig(output_name)



def build_NN(x_train, y_train, x_test, y_test, epochs,
                  model_name=None, save = True):
    '''
    This function builds the neural network model and subsequently
    fits the model for a specified number of epochs. After running it
    can be saved to a .h5 file. This model is contains convolutional
    neural network layers which analyze image data multi-class
    multilabel classifier with a sigmoid classifier with 18 output
    nodes which correspond to the 18 classes. This means that even
    though correlations between the classes of the dataset may
    exist they are ignored as they are not indicative of reality.

    **Parameters**

        x_train: *numpy.array*,*float*
            The training set of the model containing image data in
            numpy arrays. The size depends on depends on the number of
            samples but the shape of each image is (64,64,3) to match
            the input of the model.
        y_train: *numpy.array*,*int*
            The labels corresponding to the training set which are vectors
            of length 18 populated with either a 0 or 1, for all 18 classes.
        x_test: *numpy.array*, *float*
            The testing set of images which will be used to test the model.
            The size depends on the number of samples to test. But the shape of
            the containig images should be (64,64,3) like the training set.
        y_test: *numpy.array*, *int*
            The labels corresponding to the testing set images which are
            vectors of length 18 populated with either a 0 or 1, for all
            18 classes.
        model_name: *str, optional*
            The filename of the model to be saved. Should have extension h5
        epochs: *int*
            The number of times to run the data through a set of weight updates
            in the layers and back-propagation.
        save: *bool*
            Whether to saveor not. True to save as the specified model_name.

    **Returns**

        model: *Sequenctial Object*
            A sequential class that holds the layer structure and paramters
            of the model.
        history:*History Object*
            A History object that holds the weights of the trained model.

    '''


    model = Sequential()
    conv2d_1 = Conv2D(32, kernel_size= (3,3), input_shape = x_train[0].shape)
    activ = Activation('relu')
    model.add(conv2d_1) 
    #Initialize CNN with 32 62x62 inputs
    model.add(BatchNormalization())
    model.add(MaxPool2D(2,2))
    model.add(activ)
    model.add(Dropout(0.1))
    conv2d_2 = Conv2D(64, kernel_size= (3,3)) 

    model.add(conv2d_2) #Adding another CNN
    model.add(BatchNormalization())
    model.add(MaxPool2D(2,2))
    model.add(activ)
    model.add(Dropout(0.2))

    #Can add more CNNs if you want...


    model.add(Flatten())

    model.add(Dense(64))
    model.add(activ)
    model.add(BatchNormalization())
    model.add(Dropout(0.3))

    model.add(Dense(18))
    #output layer, with 18 nodes representing classes
    model.add(Activation('sigmoid'))
    model.summary()
    model.compile(loss= "binary_crossentropy",
                        optimizer = "adam",
                        metrics = ["accuracy"])
    # binary-cross entropy is loss function that poses a binary
    # classification at the output layer when fitting the data.
    # We do not use other loss functions such as categorical cross
    # -entropy because each sample may contain more than 1 label
    history = model.fit(x_train,y_train,
                        batch_size = 32,
                        epochs=epochs,
                        verbose=1,
                        validation_data=(x_test,y_test))

    #Save the model
    model_dir_name = os.path.join("New_Models", model_name)
    if model_dir_name[-3:] != ".h5":
        model_dir_name += ".h5"
    save_dir = os.getcwd()
    model_path = os.path.join(save_dir, model_dir_name)
    model.save(model_path)
    print('Saved trained model at %s ' % model_path)

    #Return the model
    return model, history


#Needed for classification
type_list = ['bug',
 'dark',
 'dragon',
 'electric',
 'fairy',
 'fighting',
 'fire',
 'flying',
 'ghost',
 'grass',
 'ground',
 'ice',
 'normal',
 'poison',
 'psychic',
 'rock',
 'steel',
 'water']


def pokemon_type_predict(image_array,trained_model, random=False, threshold=2.5):
    '''
    This function predicts the top 1 or top 2 types
    of an image based on a trained model. It decided
    based on a threshold determined from multiple runs.
    **Parameters**
        image_array: *np.array**float*
                This is an image that is turned into
                an array by array_image() function. The
                pixel data is normalized to [0,1]. Should
                be size (64,64,3).
        random: *bool*
                If this is true then the model predicts
                the type based on a random image in the
                database.
        threshold: *float*
                This is the threshold by which if the 
                probability of the second class is lower
                than the top class, it will only return
                the first prediction.
    **Returns**
        type1: *str*
                First type prediction
        type2: *str*
                Second most probable type prediction
        None: *NoneType*
                If no second type is returned


    '''

    label_p = trained_model.predict(np.array([image_array]),verbose=1)
    
    top_label, second_label =  np.argsort(label_p[0])[:-3:-1]
    if top_label > threshold * second_label:#2.5 is a threshold I picked
        return type_list[top_label], None
    top_type, second_type = type_list[top_label], type_list[second_label]
    return top_type, second_type

if __name__ == "__main__":

    cwd = os.getcwd()
    index_type_dict =  load_obj("Dictionary_index_to_type")
    index_to_one_hot = load_obj('Dictionary_index_to_onehot')
    database_dir = os.path.join(cwd,"Data/Train_test_Database")
    file_name_X = "Database_6000_X_images.npy" 
    file_name_Y = "Database_6000_Y_labels.npy"
    # Can load your own database using workflow in:
    # "Pokemon_database_pandas_manipulation.ipynb"
    relative_path1 = os.path.join(database_dir,file_name_X)
    relative_path2 = os.path.join(database_dir, file_name_Y)

    X = np.load(relative_path1, allow_pickle=True)
    Y = np.load(relative_path2, allow_pickle=True)

    #preprocess and center pics here
    processed_X = np.array([centralize_image(j) for j in X]) 

    #resize them to 100,100,3
    p_X = np.array([image_array_resize(i, (64,64,3)) for i in processed_X])

    #p_hsv_X = np.array([rgb_hsv(each_image) for each_image in p_X])
    # # UnComment if you want to train the HSV model

    #X_hsv_train, X_hsv_test, Y_train, Y_test = train_test_split(p_hsv_X,Y,
    #                                            test_size = 0.2, random_state=1)
    # # UnComment if you want to train the HSV model

    X_train, X_test, Y_train, Y_test = train_test_split(p_X,Y, test_size = 0.2,
                                                         random_state=1)
    #  Shuffle, split training and testing set by a user-defined ratio train:test.

    print(X_train.shape, Y_train.shape,X_test.shape, Y_test.shape)

    model, history = build_NN(X_train, Y_train, X_test, Y_test, model_name= "Pokemon_ML_your_trained_model.h5", epochs=10)

    plot_performance(model,history, output_name= "Pokemon_ML_your_trained_model_performance.png")


    