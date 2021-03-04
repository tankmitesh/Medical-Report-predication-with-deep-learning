# import
import tensorflow as tf
import keras
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.applications.densenet import preprocess_input
import time
import warnings
import os
import cv2
import pandas as pd
import pickle
warnings.filterwarnings("ignore")

#######################################################

# import tokenizer dict
# index to word dict
token = open('/content/token_index', 'rb')
index_word = pickle.load(token)

# word to index dict
word_index = dict((value, key) for key, value in index_word.items())

####################################################################

# import image model, chexnet model shape (None, 9, 9, 1024) last layer
image_model = tf.saved_model.load('/content/drive/MyDrive/dl case study/image_model')

# encoder model, encoder take one input as image tensor with shape (None, 9,9,1024)
# encoder model return output shape as (None, 81, 300)
encoder =  tf.saved_model.load('/content/drive/MyDrive/dl case study/encoder')

# decoder model, decoder take input as 4 different values
# dec_input = <start> token with shape (None, 1)
# encoder output = image tensor with shape (None, 81, 300)
# forward hidden  = hidden state with shape (None, 1000), 1000(defined while training)
# backward hidden = hidden state with shape (None, 1000), 1000(defined while training)
decoder = tf.saved_model.load('/content/drive/MyDrive/dl case study/decoder')

#################################################################################


def image_to_report(img1, img2) :
    
    '''
    INPUT == IMAGES OF X RAY
    OUTPUT == RETURN MEDICAL REPORT OF IMAGES
    THIS FUNCTION TAKE TWO IMAGES AND RETURN MEDICAL REPORT OF THOSE IMAGES
    '''
    
    # check exteion in input data
    extension = ['png', 'jpg', 'jpeg']
    
    ext_1 = img1.split('.')[1]
    ext_2 = img2.split('.')[1]
    
    if (ext_1 in extension) and (ext_2 in extension) :
        pass
    else:
        print('Input must be image')
    

    # load or read images
    img1 = cv2.imread(img1)
    img2 = cv2.imread(img2)
    
    # resize both images into 299,299
    img1 = cv2.resize(img1, (299,299))
    img2 = cv2.resize(img2, (299,299))
    
    # concate both image and create single image
    img_concat = cv2.hconcat([img1, img2])
    
    # resize concate image to 299,299
    img1 = tf.image.resize(img_concat, (299,299))
    
    # expand dim
    img1 = tf.expand_dims(img1, 0)
    
    # preprocess image, normalize image
    img1 = preprocess_input(img1)
    
    # convert image into (9,9,1024) shape tensor
    img_features = image_model(img1)
    
    # initialize forward
    forward      =  tf.zeros((1, 1000))
    
    # initialize backward
    backward     =  tf.zeros((1, 1000))
    
    # using encoder convert image tensor with shape (None, 9,9, 1024) to (None, 81, 300)
    img_features =  encoder(img_features)
    
    # initialize dec input
    dec_input = tf.expand_dims([word_index['<start>']], 0)
    
    text = " "
    
    # max output length is 91
    for i in range(91):
        
        # convert encoder output into text data
        predictions, forward, backward, _ = decoder(dec_input, img_features, forward, backward)

        predicted_id = np.argmax(predictions, axis = 1).ravel()[0]

        text += ' ' + index_word[predicted_id]

        if index_word[predicted_id] == '<end>':
            
            return text

        dec_input = tf.expand_dims([predicted_id], 0)
    
    # return text data without <end> token
    return text[:-5]


