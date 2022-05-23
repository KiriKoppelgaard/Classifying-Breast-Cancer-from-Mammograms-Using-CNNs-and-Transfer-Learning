import tensorflow as tf
from tensorflow import keras
import cv2
import sys, os
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D, AveragePooling2D, BatchNormalization, Activation, GlobalAveragePooling2D
from tensorflow.keras.applications import InceptionV3, EfficientNetV2M

def _parse_function(example):
    """
    A function that maps a tf record to a feature dictionary. 
    Source: https://www.kaggle.com/code/aayushkandpal/ddsm-vgg19/notebook 

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
    Starting point from: https://www.kaggle.com/code/aayushkandpal/ddsm-vgg19/notebook 

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
        # if not transfer_learning: 
        #     # downsize image to 100x100 pixels
        #     image=cv2.resize(image,(100,100))
        if transfer_learning:
            # reformat for RGB channels (since the images are b/w, we duplicate grey scale values)
            image=cv2.merge([image,image,image])
        #image # commented out from orig kaggle code 
        # append reshapes images 
        images.append(image)
        # append labels
        labels.append(image_features['label'].numpy()) # changed from 'label'
    
    return images, labels

def cnn(input_shape, conv_layers = [100], kernel_size = 3, dense_layers = [256]):
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

    model.add(BatchNormalization()) # normalise inputs 

    #add dense layers
    for dense_layer in dense_layers: 
        model.add(Dense(dense_layer, activation=tf.nn.relu)) #feed-forward layer with relu activation function (following Abdelrahman et al. 2021 using leaky relu)
        model.add(Dropout(0.2)) #randomly pruning nodes to reduce overfitting

    #output layer
    model.add(Dense(5, activation='softmax')) #output layer; five classes
              
    return model

def old_transfer_learning_model(base_model, input_shape): 
    """
    A function that defines a transfer learning model 

    Args:
        base_model (str): Which base model to use. Either 'inceptionv3' or 'efficientnetv2m'

    Returns:
        _type_: _description_
    """    
    # if we do base_model.summary() we get the details of the base_model layers 
    if base_model == 'inceptionv3':
            base_model = InceptionV3(
                input_shape=input_shape, # define input/image shape
                weights='imagenet', # include pre-trained weights from training on imagenet
                include_top=False) # leave out the top/last fully connected layer
    elif base_model == 'efficientnetv2m':
            base_model = EfficientNetV2M(
                input_shape=input_shape, # define input/image shape
                weights='imagenet', # include pre-trained weights from training on imagenet
                include_top=False) # leave out top/last fully connected layer
    else: 
        "Error: base_model must be either 'inceptionv3' or 'efficientnetv2m"
    
    model=Sequential() # define model as sequential so we can add layers sequentially 
    model.add(base_model)  # add base model

    model.add(AveragePooling2D()) # add a global spatial average pooling layer
    model.add(Dropout(0.2)) # 0.2 dropout 
    model.add(Flatten()) # flatten to prepare for fully connected layers 
    model.add(BatchNormalization()) # normalise inputs 

    # add two fully connected layers with 256 nodes, batch normalisation, and relu activation
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(100, activation='relu'))
    model.add(Dropout(0.2)) # dropout 0.2
    model.add(Dense(5,activation='softmax')) # output layer; five classes

    # freeze all convolutional base_model layers, so we only train the top layers (which were randomly initialized)
    for layer in base_model.layers:
        layer.trainable = False # keep weights from pre-training and only update new layers 

    return model

def transfer_learning_model(base_model, input_shape): 
    """
    A function that defines a transfer learning model 

    Args:
        base_model (str): Which base model to use. Either 'inceptionv3' or 'efficientnetv2m'

    Returns:
        _type_: _description_
    """    
    # if we do base_model.summary() we get the details of the base_model layers 
    if base_model == 'inceptionv3':
            base_model = InceptionV3(
                input_shape=input_shape, # define input/image shape
                weights='imagenet', # include pre-trained weights from training on imagenet
                include_top=False) # leave out the top/last fully connected layer
    elif base_model == 'efficientnetv2m':
            base_model = EfficientNetV2M(
                input_shape=input_shape, # define input/image shape
                weights='imagenet', # include pre-trained weights from training on imagenet
                include_top=False) # leave out top/last fully connected layer
    else: 
        "Error: base_model must be either 'inceptionv3' or 'efficientnetv2m"
    
    x = base_model.output
    x = AveragePooling2D()(x)  # average pooling layer
    x = Dropout(0.2)(x) # dropout
    x = Flatten()(x) # flatten to prepare for fully-connected layers
    x = BatchNormalization()(x) # batch normalisation
    x = Dense(256, activation='relu')(x) # fully-connected layer
    x = Dropout(0.2)(x) # dropout
    x = Dense(100, activation='relu')(x) # fully-connected layer
    x = Dropout(0.2)(x) # dropout
    predictions = Dense(5, activation='softmax')(x) # output layer; five classes

    # collect model
    model = Model(inputs=base_model.input, outputs=predictions)

    # freeze all layers in base_model so we only train only the top layers (randomly initialized)
    for layer in base_model.layers:
        layer.trainable = False
    

    return model
