import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import os 
from os.path import exists
import numpy as np
from sklearn.model_selection import *
from keras.utils.vis_utils import plot_model
from contextlib import redirect_stdout
import pandas as pd
from codecarbon import EmissionsTracker
import seaborn as sns
from datetime import datetime

#import functions
from src.utils import *

#Print GPU
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# load data
root_dir = os.path.abspath("")

filenames=[os.path.join(root_dir,'data','training10_0','training10_0.tfrecords'),
          os.path.join(root_dir,'data','training10_1','training10_1.tfrecords'), 
          os.path.join(root_dir,'data','training10_2','training10_2.tfrecords'),
          os.path.join(root_dir,'data','training10_3','training10_3.tfrecords'),
          os.path.join(root_dir,'data','training10_4','training10_4.tfrecords')
          ]

# empty lists
images, labels = [], []

print("loading data")
for file in filenames:
    image, label = read_data(file, transfer_learning=False)
    images.append(image)
    labels.append(label)
print("data has been loaded")

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
print("train and test split completed")

#clear up space 
del X
del y

# Reshaping the array to 4-dims so that it can work with the Keras API
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1) 
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)
x_val =  x_val.reshape(x_val.shape[0], x_val.shape[1], x_val.shape[2], 1)
input_shape = (x_train.shape[1], x_train.shape[2], x_train.shape[3])

# Making sure that the values are float so that we can get decimal points after division
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_val = x_val.astype('float32')

# Normalizing the RGB codes by dividing it to the max RGB value.
x_train /= 255
x_test /= 255
x_val /= 255

print('Number of images in x_train', x_train.shape[0])
print('Number of images in x_test', x_test.shape[0])
print('Number of images in x_val', x_val.shape[0])

print("starting model loop")
#create models for hyperparameter comparison
for model_name in ['cnn_small', 'cnn_medium', 'cnn_large']:
  #Create print
  print(model_name, 'initializing')

  #define emissionstracker
  tracker = EmissionsTracker()

  #define model
  if model_name == 'cnn_small':
    model = cnn(input_shape, conv_layers = [9], dense_layers = [16, 9])
  elif model_name == 'cnn_medium':
    model = cnn(input_shape, conv_layers = [9, 16], dense_layers = [16, 9])
  elif model_name == 'cnn_large':
    model = cnn(input_shape, conv_layers = [9, 16, 25], dense_layers = [16, 9])

  #create early stopping object 
  callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

  #compile model
  model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics = ['accuracy'])

  #save model parameters 
  with open(f'output/{model_name}/{model_name}_summary.txt', 'w') as f:
    with redirect_stdout(f):
        model.summary()

  #measure environmental impact and time
  tracker.start()
  start_time = datetime.now()

  # Fit model
  history = model.fit(x=x_train,y=y_train, epochs=100, validation_data=(x_val, y_val), callbacks=[callback])

  #save environmental impact 
  emissions: float = tracker.stop()
  end_time = datetime.now()
  path = os.path.join(root_dir,'output', 'co2emissions.csv')
  no_epochs = len(history.history['val_loss'])

  if exists(path): 
    with open(path,'a') as fd:
      fd.write(f'Emissions for {model_name}: {emissions} kg,  Duration: {end_time - start_time}, No. of epochs run: {no_epochs};')
  else: 
    with open(path, 'w') as fd:
      fd.write(f'Emissions for {model_name}: {emissions} kg,  Duration: {end_time - start_time}, No. of epochs run: {no_epochs};')

  # Evaluate model
  model.evaluate(x_test, y_test)

  #create predictions for test set 
  y_pred = model.predict(x_test, batch_size=64, verbose=1)
  y_pred_bool = np.argmax(y_pred, axis=1)

  #save classification report
  clsf_report = pd.DataFrame(classification_report(y_test, y_pred_bool, output_dict=True)).transpose()
  clsf_report.to_csv(f'output/{model_name}/{model_name}_clsf_report.csv', index= True)

  
  # Visualize history
  # Plot history: Loss
  plt.plot(np.array(history.history['val_loss'])*100, label = 'Validation Loss')
  plt.plot(np.array(history.history['loss'])*100, label = 'Training Loss')
  plt.title('Validation loss history')
  plt.ylabel('Loss value (%)')
  plt.xlabel('No. epoch')
  plt.legend(loc="upper right")
  plt.savefig(f'output/{model_name}/{model_name}_loss.jpg')
  plt.clf()

  # Plot history: Accuracy
  plt.plot(np.array(history.history['val_accuracy'])*100, label = 'Validation Accuracy')
  plt.plot(np.array(history.history['accuracy'])*100, label = 'Training Accuracy')
  plt.title('Validation accuracy history')
  plt.ylabel('Accuracy value (%)')
  plt.xlabel('No. epoch')
  plt.legend(loc="upper left")
  plt.savefig(f'output/{model_name}/{model_name}_accuracy.jpg')
  plt.clf()

  #create confusion matrix
  cm = pd.DataFrame(confusion_matrix(y_test, y_pred_bool))
  ax= plt.subplot()
  svm = sns.heatmap(cm, annot=True, fmt='g', ax=ax, cmap="Blues");  #annot=True to annotate cells, ftm='g' to disable scientific notation
  # labels, title and ticks
  ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels'); 
  ax.set_title('Confusion Matrix'); 
  ax.xaxis.set_ticklabels(['negative', 'benign calcification', 'benign mass', 'malignant calcification', 'malignant mass'], rotation = 90); 
  ax.yaxis.set_ticklabels(['negative', 'benign calcification', 'benign mass', 'malignant calcification', 'malignant mass'], rotation = 0);
  figure = svm.get_figure()
  figure.savefig(f'output/{model_name}/{model_name}_confusion_matrix.png', bbox_inches = 'tight') 

  #plot model architecture
  plot_model(model, f'output/{model_name}/{model_name}_architecture.png', show_shapes=True)

  # Predict using fitted model 
  # image_index = 2
  # plt.imshow(x_test[image_index].reshape(x_train.shape[1], x_train.shape[2]),cmap='Greys')
  # pred = model.predict(x_test[image_index].reshape(1, x_train.shape[1], x_train.shape[2], 1))
  # print(pred.argmax())
