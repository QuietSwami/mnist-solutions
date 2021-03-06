from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import numpy as np
import csv
from matplotlib import pyplot


def sequential_model(layers, learning_rate, loss_type, input_shape):
    """
        This creates a model, when given a layers dictionary, a learning rate optimizes and a loss type.

        Args:
            - layers: a dict. Each layer has the following keys:
                - type: FC, Conv, Pool;
                - If type = Conv:
                    - filters: a int, with the number of filters of this convolutional layer.
                    - size_of_filter: a tuple, that contains the size of the filter.
                    - activation: a string, with the type of activation to be used.
                    - is_input_shape: a bool, to check if the layer is the first.
                - If type = Pool:
                    - size_of_filter: a tuple, with the size of the filter to be used.
                - If type = FC:
                    - numb_of_nodes: a int, the number of nodes to be used in the layer.
                    - activation: a string, with the type of activation to be used.

            - learning_rate: a float, containing the size of the step to be used by the optimizer algorithm.
            - loss_type: a string, containing the algorithm that will calculate the loss. 
    """
    print('Creating the model...')
    model = Sequential()
    for k,v in layers.items():
        print(k)
        print(v)
        if k.startswith("Conv"):
            if v['is_input_shape'] == True:
                model.add(Conv2D(v['filter'], v['size_of_filter'], activation=v['activation'], input_shape=input_shape))
            else:
                model.add(Conv2D(v['filter'], v['size_of_filter'], activation=v['activation']))
        elif k.startswith('Pool'):
            model.add(MaxPooling2D(v['size_of_filter']))
        elif k.startswith('FC'):
            if k.endswith('1'):
                model.add(Dropout(0.25)) # To prevent overfitting.
                model.add(Flatten())
            model.add(Dense(v['numb_of_nodes'], activation=v['activation']))
    
    model.compile(loss=loss_type, optimizer=keras.optimizers.Adam(lr=learning_rate), metrics=['accuracy'])
    print('Model Created.')
    return model

if __name__ == "__main__":
    


    testName = 'cnn-keras2'
    modelIteration = '1'
    batch_size = 128
    num_classes = 10
    num_layers = 2
    num_filters1 = 32
    num_filters2 = 64
    num_nodes_fc1 = 128

    # input image dimensions
    img_rows, img_cols = 28, 28
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Shape of the Dataset
    # x_train: 60000, 28, 28
    # y_train: 60000
    # x_test: 10000, 28, 28
    # y_test: 10000.

    # The x arrays are the arrays with the images, and the y arrays have the labels.

    # Now, we need to reshape the arrays to pass through the model.
    # So we reshape the arrays from a 3D array of 60000, 28, 28, to a 4D array with a shape of 60000, 28, 28, 1. 
    print(K.image_data_format())
    if K.image_data_format() == 'channels_first':
        x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
        x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
        input_shape = (1, img_rows, img_cols)
    else:
        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    
    # Now we make sure that all the numbers in the array in the range of 0, 255
    x_train /= 255
    x_test /= 255

    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    


    layers_1 = {
        'Conv_1': {
            'filter': 32,
            'size_of_filter': (3,3),
            'activation': 'relu',
            'is_input_shape': True
        },
        'Pool_1': {
            'size_of_filter': (2,2)
        },
        'Conv_2': {    
            'filter': 64,
            'size_of_filter': (3,3),
            'activation': 'relu',
            'is_input_shape': False
        },
        'Pool_2': {
            'size_of_filter': (2,2)
        },
        'FC_1': {
            'numb_of_nodes': 128,
            'activation': 'relu'
        },
        'FC_2': {
            'numb_of_nodes': 10,
            'activation': 'softmax'
        }
    }

    layers_2 = {
        'Conv_1': {
            'filter': 2,
            'size_of_filter': (3,3),
            'activation': 'relu',
            'is_input_shape': True
        },
        'Pool_1': {
            'size_of_filter': (2,2)
        },
        'Conv_2': {    
            'filter': 4,
            'size_of_filter': (3,3),
            'activation': 'relu',
            'is_input_shape': False
        },
        'Pool_2': {
            'size_of_filter': (2,2)
        },
        'FC_1': {
            'numb_of_nodes': 128,
            'activation': 'relu'
        },
        'FC_2': {
            'numb_of_nodes': 10,
            'activation': 'softmax'
        }
    }

    layers_3 = {
        'Conv_1': {
            'filter': 4,
            'size_of_filter': (3,3),
            'activation': 'relu',
            'is_input_shape': True
        },
        'Pool_1': {
            'size_of_filter': (2,2)
        },
        'Conv_2': {    
            'filter': 8,
            'size_of_filter': (3,3),
            'activation': 'relu',
            'is_input_shape': False
        },
        'Pool_2': {
            'size_of_filter': (2,2)
        },
        'FC_1': {
            'numb_of_nodes': 128,
            'activation': 'relu'
        },
        'FC_2': {
            'numb_of_nodes': 10,
            'activation': 'softmax'
        }
    }

    layers_4 = {
        'Conv_1': {
            'filter': 8,
            'size_of_filter': (3,3),
            'activation': 'relu',
            'is_input_shape': True
        },
        'Pool_1': {
            'size_of_filter': (2,2)
        },
        'Conv_2': {    
            'filter': 16,
            'size_of_filter': (3,3),
            'activation': 'relu',
            'is_input_shape': False
        },
        'Pool_2': {
            'size_of_filter': (2,2)
        },
        'FC_1': {
            'numb_of_nodes': 128,
            'activation': 'relu'
        },
        'FC_2': {
            'numb_of_nodes': 10,
            'activation': 'softmax'
        }
    }

    layers_5 = {    
        'Conv_1': {
            'filter': 16,
            'size_of_filter': (3,3),
            'activation': 'relu',
            'is_input_shape': True
        },
        'Pool_1': {
            'size_of_filter': (2,2)
        },
        'Conv_2': {    
            'filter': 32,
            'size_of_filter': (3,3),
            'activation': 'relu',
            'is_input_shape': False
        },
        'Pool_2': {
            'size_of_filter': (2,2)
        },
        'FC_1': {
            'numb_of_nodes': 128,
            'activation': 'relu'
        },
        'FC_2': {
            'numb_of_nodes': 10,
            'activation': 'softmax'
        }
    }

    dic = {}

    results = {'batch_size': batch_size}

    for i in [2,3,5,10]:
        dic[i] = []
        for j in [layers_1, layers_2, layers_3, layers_4, layers_5]:
            model_1 = sequential_model(j, 0.01, keras.losses.categorical_crossentropy, input_shape)
            history_1 = model_1.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=i,
                    verbose=1,
                    validation_data=(x_test, y_test))
            dic[i].append([history_1])
            results['epoch_' + str(i)] = j

    with open(testName +'.json', 'w') as f:
        f.write(json.dumps(results))
        f.close()
