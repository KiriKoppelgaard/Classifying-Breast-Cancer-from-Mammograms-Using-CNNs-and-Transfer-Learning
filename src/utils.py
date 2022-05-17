#source: https://www.kaggle.com/code/aayushkandpal/ddsm-vgg19/notebook
 
import tensorflow as tf
import cv2

#Initiate variables
images=[]
labels=[]

def _parse_function(example):
    """_summary_

    Args:
        example (): _description_
        feature_dictionary (_type_): _description_

    Returns:
        _type_: _description_
    """    
    # prepare feature dictionary 
    feature_dictionary = {
        'label': tf.io.FixedLenFeature([], tf.int64),
        'label_normal': tf.io.FixedLenFeature([], tf.int64),
        'image': tf.io.FixedLenFeature([], tf.string)
        }

    parsed_example = tf.io.parse_example(example, feature_dictionary)
    print(parsed_example)
    return parsed_example


def read_data(filename):
    """
    A function to read the tfrecods in the DDSTM data set

    Args:
        filename (string): full path to tfrecord
    """    
    # Create a TFRecordDataset to read one or more TFRecord files
    full_dataset = tf.data.TFRecordDataset(filename,num_parallel_reads=tf.data.experimental.AUTOTUNE) 
    # Save in memory
    full_dataset = full_dataset.cache()

    # Print size 
    #print("Size of Training Dataset: ", len(list(full_dataset)))

    full_dataset = full_dataset.map(_parse_function, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    print("PRINT!", full_dataset)
    for image_features in full_dataset:
        image = image_features['image'].numpy()
        image = tf.io.decode_raw(image_features['image'], tf.uint8)
        image = tf.reshape(image, [299, 299])        
        image=image.numpy()
        image=cv2.resize(image,(100,100))
        image=cv2.merge([image,image,image])
        image
        images.append(image)
        labels.append(image_features['label_normal'].numpy()) # changed from 'label'

