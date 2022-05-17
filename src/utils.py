#source: https://www.kaggle.com/code/aayushkandpal/ddsm-vgg19/notebook
 
import tensorflow as tf
import cv2

def _parse_function(example):
    """
    A function that maps a tf record to a feature dictionary 

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


def read_data(filename):
    """
    A function to read the tfrecods in the DDSTM data set

    Args:
        filename (string): full path to tfrecord
    """    
    images, labels = [], []
    # Create a TFRecordDataset to read one or more TFRecord files
    full_dataset = tf.data.TFRecordDataset(filename,num_parallel_reads=tf.data.experimental.AUTOTUNE) 
    # Save in memory
    full_dataset = full_dataset.cache()

    # Print size 
    #print("Size of Training Dataset: ", len(list(full_dataset)))

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
        # downsize image to 100x100 pixels
        image=cv2.resize(image,(100,100))
        # reformat for RGB channels (since the images are b/w, we duplicate grey scale values)
        image=cv2.merge([image,image,image])
        #image # commented out from orig kaggle code 
        # append reshapes images 
        images.append(image)
        # append labels
        labels.append(image_features['label_normal'].numpy()) # changed from 'label'
    
    return images, labels