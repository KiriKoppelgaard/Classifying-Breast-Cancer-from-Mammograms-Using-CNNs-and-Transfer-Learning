import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.model_selection import KFold, StratifiedKFold
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import *
from keras.utils.vis_utils import plot_model

#import functions
from src.utils import *

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
input_shape = (x_train.shape[1], x_train.shape[2], x_train.shape[3])

# Making sure that the values are float so that we can get decimal points after division
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

# Normalizing the RGB codes by dividing it to the max RGB value.
x_train /= 255
x_test /= 255
print('Number of images in x_train', x_train.shape[0])
print('Number of images in x_test', x_test.shape[0])

#create models for hyperparameter comparison
model1 = cnn(input_shape, conv_layers = [100], dense_layers = [50])
model2 = cnn(input_shape, conv_layers = [100, 50], dense_layers = [100])
model3 = cnn(input_shape, conv_layers = [200, 100, 50], dense_layers = [250, 100])

for model in [model1]:#, model2, model3]:
  #print model parameters 
  print(model.summary())

  # Fit model: 1223 images?
  history = model.fit(x=x_train,y=y_train, epochs=1)

  # Evaluate model: 524
  results = model.evaluate(x_test, y_test)
  print("test loss, test acc:", results)

  #create predictions for test set 
  y_pred = model.predict(x_test, batch_size=64, verbose=1)
  print("y_pred has run")
  y_pred_bool = np.argmax(y_pred, axis=1)

  #print classification report
  print(classification_report(y_test, y_pred_bool))

  plot_model(model, 'model.png', show_shapes=True)

  # Predict using fitted model 
  # image_index = 2
  # plt.imshow(x_test[image_index].reshape(x_train.shape[1], x_train.shape[2]),cmap='Greys')
  # pred = model.predict(x_test[image_index].reshape(1, x_train.shape[1], x_train.shape[2], 1))
  # print(pred.argmax())

  # Visualize history
  # Plot history: Loss
  plt.plot(history.history['val_loss'])
  plt.title('Validation loss history')
  plt.ylabel('Loss value')
  plt.xlabel('No. epoch')
  plt.show()

  # Plot history: Accuracy
  plt.plot(history.history['val_accuracy'])
  plt.title('Validation accuracy history')
  plt.ylabel('Accuracy value (%)')
  plt.xlabel('No. epoch')
  plt.show()

