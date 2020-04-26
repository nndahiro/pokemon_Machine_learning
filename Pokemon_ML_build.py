import numpy as np
import matplotlib.pyplot as plt
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' #Suppress TF warning
import pickle
from sklearn.model_selection import train_test_split
#from Pokemon_database_pandas_manipulation.ipynb import load_obj
from keras.utils import np_utils
from keras.models import Sequential, load_model
from keras.layers.core import Dense, Flatten, Dropout, Activation
from keras.layers import Conv2D, MaxPool2D, BatchNormalization
import skimage.transform as transform

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


# BUILD MODEL:

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.2)

print(X_train.shape, Y_train.shape,X_test.shape, Y_test.shape)


def build_NN(x_train, y_train, x_test, y_test,
                  model_name, save = True):
    '''
    This function holds the neural network model as built by you, the user.
    After building the model, the model is then trained and saved to a .h5
    file.

    **Parameters**

        x_train: *numpy.ndarray*
            The set of training data, which should be an array of size
            (60000, 784).
        y_train: *numpy.ndarray*
            The set of labels for the corresponding training data, which
            should be an array of size (60000, 10).
        x_test: *numpy.ndarray*
            The set of testing data, which should be an array of size
            (10000, 784).
        y_test: *numpy.ndarray*
            The set of labels for the corresponding testing data, which should
            be an array of size (10000, 10).
        model_name: *str, optional*
            The filename of the model to be saved. Should have extension h5

    **Returns**

        model:
            A class object that holds the trained model.
        history:
            A class object that holds the history of the model.

    '''
    # Build the model
    # keras.model()
    # model(input layers, train_data x and train data y ), activation func
    # model.add(Dense(number_of_layers))
    for each_image in x_train:
        each_image = transform.resize(each_image, (each_image.shape[0] // 4, \
                                                each_image.shape[1] // 4), \
                                    anti_aliasing=True)
    model = Sequential()
    model.add(Conv2D(16, kernel_size= (3,3), input_shape = x_train[0].shape)) #Initialize dense NN with 784 input(28x28) image, linearized and 16 neurons
    model.add(BatchNormalization())
    model.add(MaxPool2D(2,2))
    model.add(Activation('relu'))
    model.add(Dropout(0.1))

    model.add(Conv2D(32, kernel_size= (3,3))) #Initialize dense NN with 784 input(28x28) image, linearized and 16 neurons
    model.add(BatchNormalization())
    model.add(MaxPool2D(2,2))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))

    model.add(Flatten())

    model.add(Dense(40))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))

    model.add(Dense(18)) #output layer, if I want it to classify in# what I wanted then output layer should have save length as classes I have.
    model.add(Activation('sigmoid'))
    model.summary()
    # Compile the model
    # Optimizer choose #nice, guessed it
    model.compile(loss= "binary_crossentropy",
                        optimizer = "adam",
                        metrics = ["accuracy"]) #binary-cross entropy is loss function that says minimize the error and choose the best?
                                                #metrics is what the model will calculate during training
    history = model.fit(x_train,y_train,
                        batch_size = 32,
                        epochs=10,
                        verbose=1,
                        validation_data=(x_test,y_test))

    #save model
    #Save the model
    model_dir_name = os.path.join("Models", model_name)
    save_dir = os.getcwd()
    model_path = os.path.join(save_dir, model_dir_name)
    model.save(model_path)
    print('Saved trained model at %s ' % model_path)

    #Return the model
    return model, history

# model, history = build_NN(X_train, Y_train, X_test, Y_test, model_name= "Pokemon_ML_no_Normalization.h5")

# plot_performance(model,history, output_name= "Pokemon_Model_sigmoid_no_Normalization.png")

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

poke_model_name = "Models/Pokemon_ML_no_Normalization.h5"
full_path = os.path.join(os.getcwd(),poke_model_name)

poke_model = load_model(full_path)
test = X_test[735]
y_prob = poke_model.predict(test.reshape(1,100,100,3),verbose=1) #Returns array of array
print(y_prob)
top3 = np.argsort(y_prob[0])[:-4:-1] # argsort returns index of items in descending order
print(top3) #17,12,7

for i in range(3):
    print(type_list[top3[i]])
plt.imshow(test)
plt.show()

# # print(one_hot_to_type([0,0,0,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0]))

# a = 0
# for i in Y:
#     if list(i) == [0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0]:
#         a += 1
# print(a)