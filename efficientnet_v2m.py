import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
import os
import numpy as np
from sklearn.model_selection import *
from keras.utils.vis_utils import plot_model
from contextlib import redirect_stdout
import pandas as pd

from tensorflow.keras.applications import EfficientNetV2M
from tensorflow.keras.layers import Dense, Dropout, Flatten, BatchNormalization, Activation # MaxPooling2D, AveragePooling2D

#import functions
from src.utils import *

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
    image, label = read_data(file, transfer_learning=True) 
    images.append(image)
    labels.append(label)

# flatten images and labels (so they are not nested lists)
images = [i for image in images for i in image]
labels = [l for label in labels for l in label]

# define train and test
X=np.array(images)
y=np.array(labels)

#divide data into train, test and val
x_train, x_test1, y_train, y_test1 = train_test_split(X, y, test_size=0.3, random_state=42,
                                                      shuffle=True,stratify=y)

x_val, x_test, y_val, y_test = train_test_split(x_test1, y_test1, test_size=0.3, random_state=42,
                                                shuffle=True,stratify=y_test1)

#clear up space 
del X
del y

# Reshaping the array to 4-dims so that it can work with the Keras API
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1) 
x_test = x_test.reshape(x_test.shape[0], x_train.shape[1], x_train.shape[2], 1)
x_val = x_val.reshape(x_val.shape[0], x_val.shape[1], x_val.shape[2], 1)
input_shape = (x_train.shape[1], x_train.shape[2], x_train.shape[3])

# Making sure that the values are float so that we can get decimal points after division
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_val = x_val.astype('float32')

# Normalizing the RGB codes by dividing it to the max RGB value.
x_train /= 255
x_test /= 255
x_val / 255
print('Number of images in x_train', x_train.shape[0])
print('Number of images in x_test', x_test.shape[0])
print('Number of images in x_val', x_val.shape[0])
print("Total number of images: ", x_train.shape[0] + x_test.shape[0] + x_val.shape[0])

#create model
base_model = InceptionV3(input_shape=(299,299,3), weights='imagenet', include_top=False) # include pre-trained weights from training on imagenet; we could also try to include a version without? 
inceptionv3=Sequential()
inceptionv3.add(base_model) 
# add additional layers 
inceptionv3.add(Dropout(0.2)) # 0.2 dropout  
inceptionv3.add(Flatten()) # flatten to prepare for fully connected layers 
inceptionv3.add(BatchNormalization()) 
# add two fully connected layers with 256 nodes, batch normalisation, and relu activation
inceptionv3.add(Dense(256))
inceptionv3.add(BatchNormalization())
inceptionv3.add(Activation('relu')) 
inceptionv3.add(Dropout(0.2))
inceptionv3.add(Dense(256))
inceptionv3.add(BatchNormalization())
inceptionv3.add(Activation('relu'))
inceptionv3.add(Dropout(0.2)) # dropout 0.2
inceptionv3.add(Dense(2,activation='sigmoid')) # last layer

for layer in base_model.layers:
     layer.trainable = False

inceptionv3.summary()
# inceptionv3 = InceptionV3(
#     include_top=True, # include the fully-connected layer at the top, as the last layer of the network
#     weights="imagenet", # include 
#     input_tensor=None,
#     input_shape=(None, 299, 299, 3),
#     pooling='avg', # last layer is avg pooling as in our cnn
#     classes=1000, # classes to classify; positive or negative --> needs to be 1000 if 'weights' = 'imagenet with include_top
#     classifier_activation="softmax", # top/last layer activation function; softmax 
# )

#counter
iteration = 0

for model in [inceptionv3]:
  #Create print
  iteration += 1
  print('Model', iteration, 'initializing')

  #compile model
  model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics = ['accuracy'])

  #save model parameters 
  with open(f'output/inceptionv3{iteration}_summary.txt', 'w') as f:
    with redirect_stdout(f):
        model.summary()

  # Fit model
  history = model.fit(x=x_train,y=y_train, epochs=2, validation_data=(x_val, y_val))

  # Evaluate model
  model.evaluate(x_test, y_test)

  #create predictions for test set 
  y_pred = model.predict(x_test, batch_size=64, verbose=1)
  y_pred_bool = np.argmax(y_pred, axis=1)

  #save classification report
  clsf_report = pd.DataFrame(classification_report(y_test, y_pred_bool, output_dict=True)).transpose()
  clsf_report.to_csv(f'output/model{iteration}_clsf_report.csv', index= True)

  #plot model architecture
  plot_model(model, f'output/model{iteration}_architecture.png', show_shapes=True)

  # Visualize history
  # Plot history: Loss
  plt.plot(history.history['val_loss'])
  plt.title('Validation loss history')
  plt.ylabel('Loss value')
  plt.xlabel('No. epoch')
  plt.savefig(f'output/model{iteration}_loss.png')
  plt.clf() 

  # Plot history: Accuracy
  plt.plot(history.history['val_accuracy'])
  plt.title('Validation accuracy history')
  plt.ylabel('Accuracy value (%)')
  plt.xlabel('No. epoch')
  plt.savefig(f'output/model{iteration}_accuracy.png')
  plt.clf()

  # Predict using fitted model 
  # image_index = 2
  # plt.imshow(x_test[image_index].reshape(x_train.shape[1], x_train.shape[2]),cmap='Greys')
  # pred = model.predict(x_test[image_index].reshape(1, x_train.shape[1], x_train.shape[2], 1))
  # print(pred.argmax())