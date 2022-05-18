import tensorflow as tf
import matplotlib.pyplot as plt
# Importing the required Keras modules containing model and layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D
from sklearn.metrics import classification_report
import os
import numpy as np
from sklearn.model_selection import *

#import functions
from src.utils import read_data


# load data
root_dir = os.path.abspath("")

filenames=[os.path.join(root_dir,'data','training10_0','training10_0.tfrecords')#,#'../input/ddsm-mammography/training10_0/training10_0.tfrecords',
          #os.path.join(root_dir,'data','training10_1','training10_1.tfrecords'), #'../input/ddsm-mammography/training10_1/training10_1.tfrecords',
          #os.path.join(root_dir,'data','training10_2','training10_2.tfrecords'),#'../input/ddsm-mammography/training10_2/training10_2.tfrecords',
          #os.path.join(root_dir,'data','training10_3','training10_3.tfrecords'), #'../input/ddsm-mammography/training10_3/training10_3.tfrecords',
          #os.path.join(root_dir,'data','training10_4','training10_4.tfrecords') #'../input/ddsm-mammography/training10_4/training10_4.tfrecords'
          ]

# empty lists
images, labels = [], []

for file in filenames:
    image, label = read_data(file, transfer_learning=False)
    images.append(image)
    labels.append(label)

# flatten images and labels (so they are not nested lists)
images = [i for image in images for i in image]
labels = [l for label in labels for l in label]

# define train and test
X=np.array(images)
y=np.array(labels)

print(len(images))
print(len(labels))

#divide data into train and test
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0,shuffle=True,stratify=y)
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

#(x_train_mnist, y_train_mnist), (x_test_mnist, y_test_mnist) = tf.keras.datasets.mnist.load_data()
#print(x_train_mnist.shape, x_test_mnist.shape, y_train_mnist.shape, y_test_mnist.shape)

# plot example image
image_index = 7777
print(y_train[image_index]) 
#plt.imshow(x_train[image_index], cmap='Greys')
#plt.show()


# Reshaping the array to 4-dims so that it can work with the Keras API
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1) 
x_test = x_test.reshape(x_test.shape[0], x_train.shape[1], x_train.shape[2], 1)
input_shape = (x_train.shape[1], x_train.shape[2], x_train.shape[3])

# Making sure that the values are float so that we can get decimal points after division
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

# Normalizing the RGB codes by dividing it to the max RGB value.
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print('Number of images in x_train', x_train.shape[0])
print('Number of images in x_test', x_test.shape[0])

def cnn(filters = 100, kernel_size = 3, pool_size = 2, hidden_layers = [250], activation = tf.nn.leaky_relu):
    # Creating a Sequential Model and adding the layers
    model = Sequential() #preparing for linear stack of layers
    model.add(Conv2D(filters, kernel_size=(kernel_size,kernel_size), input_shape=input_shape)) #defining number of filters and size of kernel
    model.add(MaxPooling2D(pool_size=(pool_size, pool_size))) #densing pixel information
    model.add(Flatten()) # Flattening the 2D to 1D arrays for fully connected layers
    for layer in hidden_layers: 
        model.add(Dense(layer, activation=activation)) #feed-forward layer with relu activation function (following Abdelrahman et al. 2021 using leaky relu)
        model.add(Dropout(0.2)) #randomly pruning nodes to reduce overfitting
    model.add(Dense(2,activation='sigmoid')) #feed-forward layer with softmax
    return model

model = cnn()
#print model parameters 
model.summary

# Compile model
model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])
print("model has been compiled")
# Fit model: 1223 images?
model.fit(x=x_train,y=y_train, epochs=1)
print("model has been fitted")
# Evaluate model: 524
model.evaluate(x_test, y_test)
print("model has been evaluated")

# Predict using fitted model 
# image_index = 2
# plt.imshow(x_test[image_index].reshape(x_train.shape[1], x_train.shape[2]),cmap='Greys')
# pred = model.predict(x_test[image_index].reshape(1, x_train.shape[1], x_train.shape[2], 1))
# print(pred.argmax())

#create predictions for test set 
y_pred = model.predict(x_test, batch_size=64, verbose=1)
print("y_pred has run")
y_pred_bool = np.argmax(y_pred, axis=1)

#print classification report
print(classification_report(y_test, y_pred_bool))