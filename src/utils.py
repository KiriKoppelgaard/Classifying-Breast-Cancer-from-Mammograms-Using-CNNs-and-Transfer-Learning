#source: https://www.kaggle.com/code/aayushkandpal/ddsm-vgg19/notebook
 
import tensorflow as tf
from tensorflow import keras
import cv2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D, AveragePooling2D
from sklearn.metrics import classification_report
import pandas as pd

def _parse_function(example):
    """
    A function that maps a tf record to a feature dictionary 

    Args:
        example (TFRecord): tf record, a simple format for storing a sequence of binary records

    Returns:
        dict: dictionary of features, 'label', 'label_normal', and 'image'
    """    
    # prepare feature dictionary 
    feature_dictionary = {
        'label': tf.io.FixedLenFeature([], tf.int64),
        'label_normal': tf.io.FixedLenFeature([], tf.int64),
        'image': tf.io.FixedLenFeature([], tf.string)
        }

    parsed_example = tf.io.parse_example(example, feature_dictionary)
    return parsed_example


def read_data(filename,transfer_learning=True):
    """
    A function to read the tfrecods in the DDSTM data set

    Args:
        filename (string): full path to tfrecord
    """    
    images, labels = [], []

    # Create a TFRecordDataset to read one or more TFRecord files
    full_dataset = tf.data.TFRecordDataset(filename,num_parallel_reads=tf.data.experimental.AUTOTUNE) 
    
    # Save in memory
    full_dataset = full_dataset.cache()

    #drop corrupted files 
    full_dataset = full_dataset.apply(tf.data.experimental.ignore_errors()) 

    # map feature dictionary to each tfrecord in full_dataset
    full_dataset = full_dataset.map(_parse_function, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    # for each image
    for image_features in full_dataset:
        # convert image to numpy
        image = image_features['image'].numpy()
        # convert to numeric tensor
        image = tf.io.decode_raw(image_features['image'], tf.uint8)
        # reshape 
        image = tf.reshape(image, [299, 299])        
        # make numpy again
        image=image.numpy()
        # downsize image to 100x100 pixels
        image=cv2.resize(image,(100,100))
        if transfer_learning:
            # reformat for RGB channels (since the images are b/w, we duplicate grey scale values)
            image=cv2.merge([image,image,image])
        #image # commented out from orig kaggle code 
        # append reshapes images 
        images.append(image)
        # append labels
        labels.append(image_features['label_normal'].numpy()) # changed from 'label'
    
    return images, labels

def cnn(input_shape, conv_layers = [100], kernel_size = 3, dense_layers = [250]):
    """
    A function that defines architecture for cnn 

    Args:
        input_shape: the input shape of the data, i.e. (x_train.shape[1], x_train.shape[2], x_train.shape[3])
        conv_layers: an array containing the size of each convolutional layer
        kernel_size: the size of the convolutional filter
        dense_layers:an array containing the size of each dense layer

    Returns:
        a compiled model with a given architecture
    """  
    # Creating a Sequential Model and adding the layers
    model = Sequential() #preparing for linear stack of layers

    #adding convolutional layers
    for conv_layer in conv_layers: 
        if conv_layer == conv_layers[0]:
            model.add(Conv2D(conv_layer, kernel_size=(kernel_size,kernel_size), input_shape=input_shape, activation = tf.nn.relu)) #defining number of filters and size of kernel
        else: 
                model.add(Conv2D(conv_layer, kernel_size=(kernel_size,kernel_size), activation = tf.nn.relu)) #defining number of filters and size of kernel
        
        if conv_layer != conv_layers[-1]:
            model.add(MaxPooling2D(pool_size=(2, 2))) #densing pixel information
        else: 
            model.add(AveragePooling2D(pool_size=(2, 2))) #densing pixel information

    #flatten before fully connected network
    model.add(Flatten()) # Flattening the 2D to 1D arrays for fully connected layers

    #add dense layers
    for dense_layer in dense_layers: 
        model.add(Dense(dense_layer, activation=tf.nn.relu)) #feed-forward layer with relu activation function (following Abdelrahman et al. 2021 using leaky relu)
        model.add(Dropout(0.2)) #randomly pruning nodes to reduce overfitting

    #output layer
    model.add(Dense(2, activation='sigmoid')) #feed-forward layer with softmax
              
    return model