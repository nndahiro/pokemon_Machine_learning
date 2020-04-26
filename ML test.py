'''
Software Carpentry - Neural Network Challenge

The MNIST database is a large database of handwritten digits that is commonly
used for training various image processing systems. It contains 60,000 images
of handwritten digits (along with appropriate labels) used for training a
machine learning model to classify written digits. There are also 10,000 images
and corresponding labels that can be used for testing the learned model. Each
image is represented by a 28 x 28 array of pixels.

The goal of this assignment is to use existing deep learning toolboxes to
build a neural network that can classify handwritten digits. The build_network
function below is missing code and is where you will build the structure for
your network.

Some resources that may help are listed here:
- MNIST database website - http://yann.lecun.com/exdb/mnist/
- Keras documentation - https://keras.io/
- Deep learning tutorial from 3Blue1Brown - https://www.youtube.com/watch?v=aircAruvnKk&list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi
'''
import numpy as np
import matplotlib.pyplot as plt
import os
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential, load_model
from keras.layers.core import Dense, Dropout, Activation


def show_examples(x, y, RANDOM=True, output_name="MNIST_examples.png"):
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
        plt.title("Digit: {}".format(y[i]))
        plt.xticks([])
        plt.yticks([])
    # Save figure as .png file
    fig = plt.gcf()
    fig.savefig(output_name)


def build_network(x_train, y_train, x_test, y_test,
                  model_name='keras_mnist.h5'):
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
            The filename of the model to be saved.

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
    model = Sequential()
    model.add(Dense(16, input_shape = (784,))) #Initialize dense NN with 784 input(28x28) image, linearized and 16 neurons
    model.add(Activation("sigmoid"))
    model.add(Dense(40))
    model.add(Activation("sigmoid"))
    model.add(Dense(10)) #output layer, if I want it to classify in# what I wanted then output layer should have save length as classes I have.
    model.add(Activation("sigmoid"))

    # Compile the model
    # Optimizer choose #nice, guessed it
    model.compile(loss= "categorical_crossentropy",
                        optimizer = "adam",
                        metrics = ["accuracy"]) #binary-cross entropy is loss function that says minimize the error and choose the best?
                                                # metrics is what the model will calculate during training
    history = model.fit(x_train,y_train,
                        batch_size = 32,
                        epochs=10,
                        verbose=1,
                        validation_data=(x_test,y_test))

    #save model
    # Save the model
    save_dir = os.getcwd()
    model_path = os.path.join(save_dir, model_name)
    model.save(model_path)
    print('Saved trained model at %s ' % model_path)

    # Return the model
    return model, history


def plot_performance(model, history, output_name="MNIST_performance.png"):
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
    loss, accuracy = model.evaluate(x_test, y_test, verbose=2)
    print("Test Loss:", loss)
    print("Test Accuracy:", accuracy)
    # Save figure as .png file
    fig = plt.gcf()
    fig.savefig(output_name)


if __name__ == '__main__':
    # Suppress TensorFlow warnings
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # Try '2' instead of '3'

    # Load MNIST data into appropriate training and testing variables
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Show examples if desired
    show_examples(x_train, y_train)

    # Flatten the 28 x 28 arrays of images into 1D vectors
    # x_train = x_train.reshape(60000, 784)
    # x_test = x_test.reshape(10000, 784)
    # x_train = x_train.astype('float32')
    # x_test = x_test.astype('float32')

    # # Normalize the data to help with training
    # x_train /= 255
    # x_test /= 255

    '''
    Here, we convert classes using one-hot encoding. This means that we
    convert:
    0 -> [1,0,0,0,0,0,0,0,0,0]
    1 -> [0,1,0,0,0,0,0,0,0,0]
    2 -> [0,0,1,0,0,0,0,0,0,0]
    and so on...
    '''
    n_classes = 10
    # y_train = np_utils.to_categorical(y_train, n_classes)
    # y_test = np_utils.to_categorical(y_test, n_classes)

    # Build network
    #model, history = build_network(x_train, y_train, x_test, y_test, model_name="Nelson's_MNSIT_model.h5")

    # Observe the performance of the network
    #plot_performance(model, history)
    model_name = "keras_mnist.h5"
    dire = os.getcwd()
    full_path = os.path.join(dire, model_name)
    print(type(dire))
    modelo = load_model(full_path)
    print("OK!!")

    def show_example_N(x, y, RANDOM=False, output_name="MNIST_Target_examples.png",index=0):
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
        # for i in range(9):
        #     plt.subplot(3, 3, i + 1)
        #     plt.tight_layout()
        #     plt.imshow(x[i], cmap='gray', interpolation='none')
        #     plt.title("Digit: {}".format(y[i]))
        #     plt.xticks([])
        #     plt.yticks([])
        # Save figure as .png file
        plt.subplot(3, 3, 1)
        plt.tight_layout()
        plt.imshow(x[index], cmap='gray', interpolation='none')
        plt.title("Digit: {}".format(y[index]))
        plt.xticks([])
        plt.yticks([])
        fig = plt.gcf()
        fig.savefig(output_name)
    show_example_N(x_train,y_train, index=59998)
    print(modelo.predict(x_train[59998].reshape(1,784),verbose=1))

    #x_train is a picture, converted to array by reshaping to 1 row.reshape(number_of_pics, lengthxwidth) & then .astype(float32)
    #TODO make a labelled dataset by this weekend
    #Probably a pandas.df of image and pokemon types
    #preprocessing images to same size
    '''
    for i in tqdm(range(data.shape[0])): # data here is the long list of labelled data. The samples here will be pictures of pokemon,not ht epokemon themeselves. e.g: multiple charizards
        path = "where images are"
        image = image.open(path)
        image = image_to_array(image, normalized = True)

        all_images.append(image)
    '''
        

