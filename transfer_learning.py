'''
Script to run the transfer learning models, inceptionv3 and efficientnetv2s. 

First, the data is prepared. 
Then, we loop through the two models. 
For the first couple of epochs, the entire base model is frozen (i.e. non-trainable) to train 
the new fully-connected top layers, so the weights are not entirely random when we start fine-tuning. 
Afterwards, the two top blocks of the base models are unfrozen (i.e made trainable), and the fine-tuning begins. 

After fine-tuning, the models are evaluated and relevant metrics and plots are saved in the output folder. 

Transfer learning and fine-tuning code sources
- https://keras.io/guides/transfer_learning/
- https://keras.io/api/applications/#usage-examples-for-image-classification-models
- https://keras.io/api/applications/#inceptionv3 (freezing blocks + low learning rate)
- https://stackabuse.com/image-recognition-in-python-with-tensorflow-and-keras/

'''

import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import os
from os.path import exists
import numpy as np
from sklearn.model_selection import *
from keras.utils.vis_utils import plot_model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import load_model
from contextlib import redirect_stdout
import pandas as pd
from codecarbon import EmissionsTracker
import seaborn as sns
from datetime import datetime

# import functions
from src.utils import *

# print how many GPUs are available
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# load data
root_dir = os.path.abspath("")

filenames=[os.path.join(root_dir,'data','training10_0','training10_0.tfrecords'),
          os.path.join(root_dir,'data','training10_1','training10_1.tfrecords'), 
          os.path.join(root_dir,'data','training10_2','training10_2.tfrecords'),
          os.path.join(root_dir,'data','training10_3','training10_3.tfrecords'), 
          os.path.join(root_dir,'data','training10_4','training10_4.tfrecords')
          ]

images, labels = [], []

print("loading data")
for file in filenames:
    image, label = read_data(file, transfer_learning=True) # tl=True: duplicate image shape so we have three channels 
    images.append(image)
    labels.append(label)
print("data has been loaded")

# flatten data (images and labels), so they are not nested lists
images = [i for image in images for i in image]
labels = [l for label in labels for l in label]

# define train and test
X=np.array(images)
y=np.array(labels)

# divide data into train, test and val
x_train, x_test1, y_train, y_test1 = train_test_split(X, y, test_size=0.4, random_state=3,
                                                      shuffle=True,stratify=y)

x_val, x_test, y_val, y_test = train_test_split(x_test1, y_test1, test_size=0.5, random_state=3,
                                                shuffle=True,stratify=y_test1)
print("train and test split completed")
# clear up space 
del X
del y

# define input shape for conv
input_shape = (x_train.shape[1], x_train.shape[2], x_train.shape[3])

# making sure that the values are float so we can get decimal points after division
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_val = x_val.astype('float32')

# normalizing the RGB codes by dividing with the max RGB value
x_train /= 255
x_test /= 255
x_val /= 255

# print info about pre-processed data
print('Number of images in x_train', x_train.shape[0], ', x_train shape is', x_train.shape)
print('Number of images in x_test', x_test.shape[0], ', x_test shape is', x_test.shape)
print('Number of images in x_val', x_val.shape[0], ', x_val shape is', x_val.shape)
print('Total number of images:', x_train.shape[0] + x_test.shape[0] + x_val.shape[0])

# prepare names of base models to loop through
base_models = ['inceptionv3','efficientnetv2s'] 


print("starting model loop")
# fine-tune and evaluate base models
for base_model in base_models: 
  # load current base model (with non-trainable base_model layers)
  if base_model == 'inceptionv3':
    model = transfer_learning_model('inceptionv3', input_shape) 
  elif base_model == 'efficientnetv2s':
    model = transfer_learning_model('efficientnetv2s', input_shape) 
  
  # print model initialisation
  print(base_model, 'initializing')

  # define emissions tracker
  tracker = EmissionsTracker()

  # define callback (for early stopping)
  #callback = EarlyStopping(monitor='val_loss', patience=10)
  es_frozen = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=20)
  mc_frozen = ModelCheckpoint(f'output/{base_model}/best_model_frozen.h5', monitor='val_loss', mode='min', verbose=1, save_best_only=True)

  es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=50)
  mc = ModelCheckpoint(f'output/{base_model}/best_model.h5', monitor='val_loss', mode='min', verbose=1, save_best_only=True)

  # compile model
  model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics = ['accuracy'])

  # save model parameters with frozen base_model
  with open(f'output/{base_model}/{base_model}_frozen_summary.txt', 'w') as f:
    with redirect_stdout(f):
        model.summary()

  #measure environmental impact and time
  tracker.start()
  start_time = datetime.now()

  # fit initial model (train on a few epochs before unfreezing two top blocks of base model for fine-tuning)
  history = model.fit(x=x_train,y=y_train, epochs=20, validation_data=(x_val, y_val),callbacks=[es_frozen, mc_frozen],verbose=1)
  print("pre-training completed for", base_model)

  # load best frozen model for finetuning
  model = load_model(f'output/{base_model}/best_model_frozen.h5')

  # unfreeze base model for finetuning
  for layer in model.layers:
    layer.trainable = True
  
  # recompile the model for modifications to take effect - with low learning rate
  model.compile(keras.optimizers.Adam(0.0001), loss='sparse_categorical_crossentropy', metrics = ['accuracy']) 

  # save model parameters with unfrozen top blocks of base_model
  with open(f'output/{base_model}/{base_model}_finetuning_summary.txt', 'w') as f:
    with redirect_stdout(f):
        model.summary()   
  
  # fine-tune model (training two top blocks of base model + fully-connected layers) 
  history_finetuning = model.fit(x=x_train,y=y_train, epochs=200, validation_data=(x_val, y_val),callbacks=[es, mc],verbose=1) #, callbacks=[callback])
  print("finetuning completed for", base_model)

  #save environmental impact + no. of epochs
  emissions: float = tracker.stop()
  end_time = datetime.now()
  path = os.path.join(root_dir,'output', 'co2emissions.csv')
  no_epochs = len(history.history['val_loss'])

  if exists(path): 
    with open(path,'a') as fd:
      fd.write(f'Emissions for {base_model}: {emissions} kg,  Duration: {end_time - start_time}, No. of epochs run: {no_epochs};')
  else: 
    with open(path, 'w') as fd:
      fd.write(f'Emissions for {base_model}: {emissions} kg,  Duration: {end_time - start_time}, No. of epochs run: {no_epochs};')

  # load the saved model
  model = load_model(f'output/{base_model}/best_model.h5')
  # evaluate the model
  _, train_acc = model.evaluate(x_train, y_train, verbose=0)
  _, test_acc = model.evaluate(x_test, y_test, verbose=0)
  print('Train: %.3f, Test: %.3f' % (train_acc, test_acc))    
  # evaluate model  
  #model.evaluate(x_test, y_test)

  # create predictions for test set 
  y_pred = model.predict(x_test, batch_size=64, verbose=1)
  y_pred_bool = np.argmax(y_pred, axis=1)

  # save classification report
  clsf_report = pd.DataFrame(classification_report(y_test, y_pred_bool, output_dict=True)).transpose()
  clsf_report.to_csv(f'output/{base_model}/{base_model}_clsf_report.csv', index= True)

  # save history
  hist_df = pd.DataFrame(history.history)
  hist_df.to_csv(f'output/{base_model}/{base_model}_history.csv', index= True)
          
  # save history
  hist_finetuning_df = pd.DataFrame(history_finetuning.history)
  hist_finetuning_df.to_csv(f'output/{base_model}/{base_model}_history_finetuning.csv', index= True)

# plot frozen history: loss
  plt.plot(np.array(history.history['val_loss']), label = 'Validation Loss')
  plt.plot(np.array(history.history['loss']), label = 'Training Loss')
  plt.title('Validation loss history')
  plt.ylabel('Loss value')
  plt.xlabel('No. epoch')
  plt.legend(loc="upper right")
  plt.savefig(f'output/{base_model}/{base_model}_frozen_loss.jpg')
  plt.clf()

  # plot frozen history: accuracy
  plt.plot(np.array(history.history['val_accuracy'])*100, label = 'Validation Accuracy')
  plt.plot(np.array(history.history['accuracy'])*100, label = 'Training Accuracy')
  plt.title('Validation accuracy history')
  plt.ylabel('Accuracy value (%)')
  plt.xlabel('No. epoch')
  plt.legend(loc="upper left")
  plt.savefig(f'output/{base_model}/{base_model}_frozen_accuracy.jpg')
  plt.clf()

  # plot finetuning history: loss
  plt.plot(np.array(history_finetuning.history['val_loss']), label = 'Validation Loss')
  plt.plot(np.array(history_finetuning.history['loss']), label = 'Training Loss')
  plt.title('Validation loss history')
  plt.ylabel('Loss value')
  plt.xlabel('No. epoch')
  plt.legend(loc="upper right")
  plt.savefig(f'output/{base_model}/{base_model}_loss.jpg')
  plt.clf()

  # plot finetuning history: accuracy
  plt.plot(np.array(history_finetuning.history['val_accuracy'])*100, label = 'Validation Accuracy')
  plt.plot(np.array(history_finetuning.history['accuracy'])*100, label = 'Training Accuracy')
  plt.title('Validation accuracy history')
  plt.ylabel('Accuracy value (%)')
  plt.xlabel('No. epoch')
  plt.legend(loc="upper left")
  plt.savefig(f'output/{base_model}/{base_model}_accuracy.jpg')
  plt.clf()

  # create confusion matrix
  cm = pd.DataFrame(confusion_matrix(y_test, y_pred_bool))
  ax= plt.subplot()
  svm = sns.heatmap(cm, annot=True, fmt='g', ax=ax, cmap="Blues");  
  ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels'); 
  ax.set_title('Confusion Matrix'); 
  ax.xaxis.set_ticklabels(['negative', 'benign calcification', 'benign mass', 'malignant calcification', 'malignant mass'], rotation = 90); 
  ax.yaxis.set_ticklabels(['negative', 'benign calcification', 'benign mass', 'malignant calcification', 'malignant mass'], rotation = 0);
  figure = svm.get_figure()
  figure.savefig(f'output/{base_model}/{base_model}_confusion_matrix.png', bbox_inches = 'tight') 

  print("Evaluation and plotting completed for", base_model)

  #pd.DataFrame(history_finetuning.history).plot()
  #plt.savefig(f'output/{base_model}/{base_model}_all_in_one_finetuning.jpg')


  # plot model architecture
  #plot_model(model, f'output/{base_model}/{base_model}_architecture.png', show_shapes=True)
