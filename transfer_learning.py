import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
import os
import numpy as np
from sklearn.model_selection import *
from keras.utils.vis_utils import plot_model
from contextlib import redirect_stdout
import pandas as pd

from tensorflow.keras.applications import InceptionV3
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

print('x_train.shape: ', x_train.shape, 'x_test.shape', x_test.shape, 'x_val.shape', x_val.shape)
input_shape = (x_train.shape[1], x_train.shape[2], x_train.shape[3])

# Making sure that the values are float so that we can get decimal points after division
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_val = x_val.astype('float32')

# Normalizing the RGB codes by dividing it to the max RGB value.
x_train /= 255
x_test /= 255
x_val /= 255
print('Number of images in x_train', x_train.shape[0], ', x_train shape is', x_train.shape)
print('Number of images in x_test', x_test.shape[0], ', x_test shape is', x_test.shape)
print('Number of images in x_val', x_val.shape[0], ', x_val shape is', x_val.shape)
print('Total number of images:', x_train.shape[0] + x_test.shape[0] + x_val.shape[0])

#create model
inceptionv3 = transfer_learning_model('inceptionv3') 
efficientnetv2 = transfer_learning_model('efficientnetv2m') 

# Print layers 
inceptionv3.summary()
efficientnetv2.summary()

# inceptionv3 = InceptionV3( ## if we do a summary of this, we get the layers of inceptionv3
#     include_top=True, # include the fully-connected layer at the top, as the last layer of the network
#     weights="imagenet", # include 
#     input_tensor=None,
#     input_shape=(None, 299, 299, 3),
#     pooling='avg', # last layer is avg pooling as in our cnn
#     classes=1000, # classes to classify; positive or negative --> needs to be 1000 if 'weights' = 'imagenet with include_top
#     classifier_activation="softmax", # top/last layer activation function; softmax 
# )

#counter
models = ['inceptionv3'] #, 'efficientnetv2m']

for base_model in models: 
  #Create print
  if base_model == 'inceptionv3':
    model = transfer_learning_model('inceptionv3') 
  elif base_model == 'efficientnetv2m':
    model = transfer_learning_model('efficientnetv2m') 
  
  print(base_model, 'initializing')

  #compile model (where base_model layers are non-trainable)
  model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics = ['accuracy'])

  #save model parameters 
  with open(f'output/{base_model}_summary.txt', 'w') as f:
    with redirect_stdout(f):
        model.summary()


  # Fit initial model (train on a few epochs before unfreezing top layers of base model for fine-tuning)
  history = model.fit(x=x_train,y=y_train, epochs=5, validation_data=(x_val, y_val))

  ############## IMPLEMENT THIS ##############
  # first: train only the top layers (which were randomly initialized)
  # i.e. freeze all convolutional InceptionV3 layers
  # we chose to train the top 2 inception blocks, i.e. we will freeze
  # the first 249 layers and unfreeze the rest:
  for layer in model.layers[:249]:
    layer.trainable = False
  for layer in model.layers[249:]:
    layer.trainable = True

  # we need to recompile the model for these modifications to take effect
  # we use SGD with a low learning rate
  from tensorflow.keras.optimizers import SGD
  model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy')

  # we train our model again (this time fine-tuning the top 2 inception blocks
  # alongside the top Dense layers
  model.fit(...)

  ############## IMPLEMENT THIS ############## https://keras.io/api/applications/#usage-examples-for-image-classification-models
  # https://medium.com/analytics-vidhya/transfer-learning-using-inception-v3-for-image-classification-86700411251b
  # Evaluate model  
  model.evaluate(x_test, y_test)

  #create predictions for test set 
  y_pred = model.predict(x_test, batch_size=64, verbose=1)
  y_pred_bool = np.argmax(y_pred, axis=1)

  #save classification report
  clsf_report = pd.DataFrame(classification_report(y_test, y_pred_bool, output_dict=True)).transpose()
  clsf_report.to_csv(f'output/{base_model}_clsf_report.csv', index= True)

  #plot model architecture
  plot_model(model, f'output/{base_model}_architecture.png', show_shapes=True)

  # Visualize history
  # Plot history: Loss
  plt.plot(np.array(history.history['val_loss'])*100, label = 'Validation Accuracy')
  plt.plot(np.array(history.history['loss'])*100, label = 'Training Accuracy')
  plt.title('Validation accuracy history')
  plt.ylabel('Accuracy value (%)')
  plt.xlabel('No. epoch')
  plt.legend(loc="upper right")
  plt.savefig(f'output/{base_model}_loss.jpg')
  plt.clf()

  # Plot history: Accuracy
  plt.plot(np.array(history.history['val_accuracy'])*100, label = 'Validation Accuracy')
  plt.plot(np.array(history.history['accuracy'])*100, label = 'Training Accuracy')
  plt.title('Validation accuracy history')
  plt.ylabel('Accuracy value (%)')
  plt.xlabel('No. epoch')
  plt.legend(loc="upper left")
  plt.savefig(f'output/{base_model}_accuracy.jpg')
  plt.clf()


## https://keras.io/api/applications/#usage-examples-for-image-classification-models