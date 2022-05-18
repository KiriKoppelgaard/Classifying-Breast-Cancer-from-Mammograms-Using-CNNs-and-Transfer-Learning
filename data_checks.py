import tensorflow as tf
import matplotlib.pyplot as plt
# Importing the required Keras modules containing model and layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D
from sklearn.metrics import classification_report
import os
import numpy as np
from sklearn.model_selection import *



# load data
root_dir = os.path.abspath("")

test_images = np.load(os.path.join(root_dir,'data','test10_data','test10_data.npy')) 
test_labels = np.load(os.path.join(root_dir,'data','test10_labels.npy'))
val_images = np.load(os.path.join(root_dir, 'data','cv10_data','cv10_data.npy'))
val_labels = np.load(os.path.join(root_dir,'data','cv10_labels.npy'))

#print(test_images.shape)

#print(type(test_images))

#print(test_labels[0])

print("test_labels")
print(np.unique(test_labels))
print(np.bincount(test_labels))
print("val_labels")
print(np.unique(val_labels))
print(np.bincount(val_labels))


image_index = 1
print("test label: ", test_labels[image_index]) # The label is 8
plt.imshow(test_images[image_index], cmap='Greys')
plt.show()


image_index = [100,101,102,103] 
for index in image_index:
    print("index is", index)
    print("test label: ", test_labels[index]) # The label is 8
    plt.imshow(test_images[index], cmap='Greys')
    plt.show()

for index in image_index:
    print("index is", index)
    print("test label: ", val_labels[index]) # The label is 8
    plt.imshow(val_images[index], cmap='Greys')
    plt.show()

test_images_0 = []
for index in range(0,len(test_labels)):
    if test_labels[index] == 0:
        test_images_0.append(test_images[index])

val_images_0 = []
for index in range(0,len(val_labels)):
    if val_labels[index] == 0:
        val_images_0.append(val_images[index])

np.array_equal(test_images_0, val_images_0)
(test_images_0==val_images_0)
#Check whether it is the same positive images in test and val
#Should we just combine them? 
#Plot some stuff 
#Find plot with accuracy and *stuff* over time and implement it 

'''
# load data
root_dir = os.path.abspath("")

filenames=[os.path.join(root_dir,'data','training10_0','training10_0.tfrecords'),#'../input/ddsm-mammography/training10_0/training10_0.tfrecords',
          os.path.join(root_dir,'data','training10_1','training10_1.tfrecords'), #'../input/ddsm-mammography/training10_1/training10_1.tfrecords',
          os.path.join(root_dir,'data','training10_2','training10_2.tfrecords'),#'../input/ddsm-mammography/training10_2/training10_2.tfrecords',
          os.path.join(root_dir,'data','training10_3','training10_3.tfrecords'), #'../input/ddsm-mammography/training10_3/training10_3.tfrecords',
          os.path.join(root_dir,'data','training10_4','training10_4.tfrecords') #'../input/ddsm-mammography/training10_4/training10_4.tfrecords'
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
'''