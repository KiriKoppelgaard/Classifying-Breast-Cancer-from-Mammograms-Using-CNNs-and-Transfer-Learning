import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.model_selection import KFold, StratifiedKFold
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import *

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
X=np.array(images)[0:1000]
y=np.array(labels)[0:1000]

print(len(images))
print(len(labels))

#divide data into train and test
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0,shuffle=True,stratify=y)
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

# plot example image
image_index = 7
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

#create model
model = cnn(input_shape)

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


#######Cross Validation
kfold = KFold(n_splits=5, shuffle=True)

# K-fold Cross Validation model evaluation
fold_no = 1
batch_size = 100
no_epochs =  1
acc_per_fold, loss_per_fold = [], []

for train, test in kfold.split(x_train, y_train):

  # Define the model architecture
  model = cnn(input_shape = input_shape)

  # Generate a print
  print(f'Training for fold {fold_no} ...')

  # Fit data to model
  history = model.fit(x_train[train], y_train[train],
              batch_size=batch_size,
              epochs=no_epochs,
              verbose=10)

  # Generate generalization metrics
  scores = model.evaluate(x_train[test], y_train[test], verbose=0)
  print(f'Score for fold {fold_no}: {model.metrics_names[0]} of {scores[0]}; {model.metrics_names[1]} of {scores[1]*100}%')
  acc_per_fold.append(scores[1] * 100)
  loss_per_fold.append(scores[0])

  # Increase fold number
  fold_no = fold_no + 1

#print outcome
print('acc_per_fold', acc_per_fold)
print('loss_per_fold', loss_per_fold)

print('mean accuracy', np.mean(np.array(acc_per_fold)))
print('mean loss', np.mean(np.array(loss_per_fold)))

