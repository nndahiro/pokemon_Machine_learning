import numpy as np
import matplotlib.pyplot as plt
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' #Suppress TF warning
import pickle
from sklearn.model_selection import train_test_split
#from Pokemon_database_pandas_manipulation.ipynb import load_obj
from keras.metrics import categorical_accuracy
from keras.utils import np_utils
from keras.models import Sequential, load_model, Model
from keras.layers.core import Dense, Flatten, Dropout, Activation
from keras.layers import Conv2D, MaxPool2D, BatchNormalization, advanced_activations
import skimage.transform as transform
from copy import deepcopy
from Image_processing import image_array_resize, centralize_image

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
    it returns a string of a list
    '''
    indexer = get_key(one_h, index_to_one_hot)
    if as_string:
        return f"{index_type_dict.get(indexer)}"
    return index_type_dict.get(indexer)

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
    fig = plt.gcf()
    fig.savefig(output_name)


def show_examples(x, y,output_name,RANDOM=True):
    '''
    Save an image containing an assortment of nine training images along with
    the corresponding labels.

    **Parameters**

        x: *numpy.ndarray*
            A set of training data.
        y: *numpy.ndarray*
            A set of labels for the corresponding training data.
        RANDOM: *bool, optional*
            A flag to decide if the selected images will be random or not. If
            changed to False, the image will show the first nine examples from
            the dataset.
        output_name: *str, optional*
            The filename of the output image.

    **Returns**

        None
    '''
    # Shuffle the training data if desired
    if RANDOM:
        p = np.random.permutation(len(x))
        x, y = x[p], y[p]
    # Plot figure using matplotlib
    plt.figure()
    for i in range(9):
        plt.subplot(3, 3, i + 1)
        plt.tight_layout()
        plt.imshow(x[i], cmap='gray', interpolation='none')
        plt.title("type: {}".format(one_hot_to_type(y[i])))
        plt.xticks([])
        plt.yticks([])
    # Save figure as .png file
    fig = plt.gcf()
    fig.savefig(output_name)




index_type_dict =  load_obj("Dictionary_index_to_type")
index_to_one_hot = load_obj('Dictionary_index_to_onehot')
database_dir = "C:/Users/Nelson/Desktop/Software carp/Final project/Data/Train_test_Database"
file_name_X = "Database_6000_X_images.npy"
file_name_Y = "Database_6000_Y_labels.npy"
relative_path1 = os.path.join(database_dir,file_name_X)
relative_path2 = os.path.join(database_dir, file_name_Y)

X = np.load(relative_path1, allow_pickle=True)
Y = np.load(relative_path2, allow_pickle=True)


#show_examples(X,Y, output_name="pOKEMON_images.png") # Randomly show examples
# plt.imshow(X[5000])
# plt.show()
processed_X = np.array([centralize_image(j) for j in X]) #preprocess and center pics here
#print(processed_X.reshape(X.shape[0],100,100,3))
# plt.imshow(processed_X[5000])
# plt.show()

p_X = np.array([image_array_resize(i, (64,64,3)) for i in processed_X]) #resize them to 100,100,3
# plt.imshow((p_X[5000]))
# # plt.show()
# # print(p_X.shape)
# BUILD MODEL:

X_train, X_test, Y_train, Y_test = train_test_split(p_X,Y, test_size = 0.2)


print(X_train.shape, Y_train.shape,X_test.shape, Y_test.shape)


def build_NN(x_train, y_train, x_test, y_test,
                  model_name=None, epochs, save = True):
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

        model: *Sequenctial() Object*
            A sequential class that holds the layer structure and paramters
            of the model.
        history:*History Object*
            A History object that holds the weights of the trained model.

    '''
    # for each_image in x_train:#blur
    #     each_image = transform.resize(each_image, (each_image.shape[0] // 4, \
    #                                             each_image.shape[1] // 4), \
    #                                 anti_aliasing=True)
    conv_layers = []
    model = Sequential()
    conv2d_1 = Conv2D(32, kernel_size= (3,3), input_shape = x_train[0].shape)
    activ = Activation('relu')
    model.add(conv2d_1) 
    #Initialize dense CNN with 32 62x62 inputs
    conv_layers.append(conv2d_1)
    model.add(BatchNormalization())
    model.add(MaxPool2D(2,2))
    model.add(activ)
    model.add(Dropout(0.1))
    conv2d_2 = Conv2D(64, kernel_size= (3,3)) 

    model.add(conv2d_2) #Initialize dense CNN
    conv_layers.append(conv2d_2)
    model.add(BatchNormalization())
    model.add(MaxPool2D(2,2))
    model.add(activ)
    model.add(Dropout(0.2))

    conv2d_3 = Conv2D(64, kernel_size= (3,3)) 

    model.add(conv2d_3) #Initialize CNN
    conv_layers.append(conv2d_3)
    model.add(BatchNormalization())
    model.add(MaxPool2D(2,2))
    model.add(activ)
    model.add(Dropout(0.2))

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
    # We do not use other loss functions because each sample
    # may contain more than 1 label
    history = model.fit(x_train,y_train,
                        batch_size = 32,
                        epochs=epochs,
                        verbose=1,
                        validation_data=(x_test,y_test))

    #Save the model
    model_dir_name = os.path.join("Models", model_name)
    save_dir = os.getcwd()
    model_path = os.path.join(save_dir, model_dir_name)
    model.save(model_path)
    print('Saved trained model at %s ' % model_path)

    #Return the model
    return model, history

# model, history = build_NN(X_train, Y_train, X_test, Y_test, model_name= "Pokemon_ML_image_center_croprelu_3Conv.h5", epochs=10)

# plot_performance(model,history, output_name= "Pokemon_ML_image_center_croprelu_3Conv_performance.png")

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

poke_model_name = "Models/Pokemon_ML_image_center.h5"
full_path = os.path.join(os.getcwd(),poke_model_name)

poke_model = load_model(full_path)
# test = p_X[5000]
# #print(poke_model.summary())
# #print(test)
# plt.imshow(test)
# plt.show()
# y_prob = poke_model.predict(test.reshape(1,100,100,3),verbose=1) #Returns array of array
# print(y_prob)
# top3 = np.argsort(y_prob[0])[:-4:-1] # argsort returns index of items in descending order
# print(top3) #17,12,7

# for i in range(3):
#     print(type_list[top3[i]])

def pokemon_type_predict(image_array, random=False, threshold=2.5):# add random feature here
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
    # tester = p_X[np.random.randint(0,4000)] #
    # testee = deepcopy(tester) #
    # plt.imshow(testee)#
    # plt.show() #
    #image_array reshaped to right shape.
    label_p = poke_model.predict(np.array([image_array]),verbose=1)
    
    top_label, second_label =  np.argsort(label_p[0])[:-3:-1]
    if top_label > threshold* second_label:#2.5 is a threshold I picked
        return type_list[top_label], None
    top_type, second_type = type_list[top_label], type_list[second_label]
    return top_type, second_type

# pred = np.array(poke_model.predict(X_test, batch_size=512))
# #print(((categorical_accuracy(Y_test, pred))) * 1)
# print(poke_model.layers[0])
# #SEEING IN LAYERS
# intermediate_layer_model = Model(inputs=poke_model.input,
#                                  outputs=poke_model.layers[0].output)
# intermediate_output = intermediate_layer_model.predict(p_X[73].reshape(1,64,64,3))
# print(intermediate_output.shape)
# plt.imshow(intermediate_output[0,:,:,30])
# plt.show()
# # type1, type2 = pokemon_type_predict(test)
# print(type1, type2)

# plt.imshow(test)
# plt.show()

# # print(one_hot_to_type([0,0,0,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0]))

# a = 0
# for i in Y:
#     if list(i) == [0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0]:
#         a += 1
# print(a)