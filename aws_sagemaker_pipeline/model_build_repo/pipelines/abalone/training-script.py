#!/usr/bin/env python
# coding: utf-8

import subprocess
import sys
subprocess.check_call([sys.executable, "-m", "pip", "install", "matplotlib"])
import numpy as np
import pandas as pd 
import random
import os
import sys
import time
import tensorflow.keras as keras
import tensorflow as tf
import tensorflow.keras.applications.xception as xception
import re
import boto3
import sagemaker.amazon.common as smac
import sagemaker
import argparse
import matplotlib.pyplot as plt
import tensorflow.python.keras.applications.xception as xception

from tensorflow.keras.optimizers import Adam
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Activation    
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.preprocessing.image import load_img
from sagemaker.tensorflow import TensorFlow
from PIL import Image
from tensorflow.python.keras.layers import Input, Conv2D, Dense, Flatten, MaxPooling2D, Input, GlobalAveragePooling2D
from tensorflow.python.keras.models import Model, Sequential
from tensorflow.python.keras.preprocessing import image
from tensorflow.python.keras.layers import Lambda
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.utils import class_weight

print('Imports successful!')

if __name__ == "__main__":
    
    # Parse the parameters
    parser = argparse.ArgumentParser()     
    parser.add_argument('--epochs', type=int, default=9)
    parser.add_argument('--learning-rate', type=float, default=0.006)
    parser.add_argument('--batch-size', type=int, default=94)
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--image_width', type=int, default = 320)
    parser.add_argument('--image_height', type=int, default = 320)
    parser.add_argument('--image_channels', type=int, default = 3)

    args, _ = parser.parse_known_args()

    epochs = args.epochs
    learning_rate = args.learning_rate
    batch_size = args.batch_size
    model_dir = args.model_dir
    image_width = args.image_width
    image_height = args.image_height
    image_channels = args.image_channels
    

    print('Parameters passed to training script: ')
    print('args.epochs : ', epochs)
    print('args.learning-rate : ', learning_rate)
    print('args.batch-size : ', batch_size)
    print('args.model-dir : ', model_dir)
    print('args.image_width : ', image_width)
    print('args.image_height : ', image_height)
    print('args.image_channels : ', image_channels)

    # S3 bucket where the pretrained Xception Weights are stored
    model_bucket_name = 'abla-garbage-classification-model'

    #S3 bucket where the training data is saved
    training_bucket_name = "abla-garbage-classification-data"
    
    #Name of the folder where the training images will be copied locally and is also the name of the folder 
    #in the s3 where the images are saved
    train_local_folder_name = 'train-and-validate'
    
    #Path where the training images will be copied locally
    train_input_path = './' + train_local_folder_name     
        
    #Dictionary to save our 12 classes
    categories = {0: 'battery', 1: 'biological', 2: 'brown-glass', 3: 'cardboard', 4: 'clothes', 5: 'green-glass',
                  6: 'metal', 7: 'paper', 8: 'plastic', 9: 'shoes', 10: 'trash', 11: 'white-glass'}
    
    
    #Functions to read and write from S3 Buckets
    def download_from_s3(filename, bucket, key):
        with open(filename,'wb') as f:
            return boto3.Session().resource('s3').Bucket(bucket).Object(key).download_fileobj(f)

    def write_to_s3(filename, bucket, key):
        with open(filename,'rb') as f: # Read in binary mode
            return boto3.Session().resource('s3').Bucket(bucket).Object(key).upload_fileobj(f)
    
    # Add class name prefix to filename. So for example "/paper104.jpg" become "paper/paper104.jpg"
    def add_class_name_prefix(df, col_name):
        df[col_name] = df[col_name].apply(lambda x: x[:re.search("\d",x).start()] + '/' + x)
        return df

    def downloadDirectoryFroms3(bucketName, remoteDirectoryName):
        s3_resource = boto3.resource('s3')
        bucket = s3_resource.Bucket(bucketName) 
        for obj in bucket.objects.filter(Prefix = remoteDirectoryName):
            if not os.path.exists(os.path.dirname(obj.key)):
                os.makedirs(os.path.dirname(obj.key))
            bucket.download_file(obj.key, obj.key) # save to same path
    
    downloadDirectoryFroms3(training_bucket_name, train_local_folder_name) 
        
    #We want to create a data frame that has in one column the filenames of all our images and in 
    #the other column the corresponding category. We Open the directories in the dataset one by one, 
    #save the filenames in the filenames_list and add the corresponding category in the categories_list
    filenames_list = []
    categories_list = []

    for category in categories:
        filenames = os.listdir(train_input_path + '/' + categories[category])

        filenames_list = filenames_list  +filenames
        categories_list = categories_list + [category] * len(filenames)

    df = pd.DataFrame({
        'filename': filenames_list,
        'category': categories_list
    })

    df = add_class_name_prefix(df, 'filename')

    # Shuffle the dataframe
    df = df.sample(frac=1).reset_index(drop=True)

    print('number of elements = ' , len(df))

    print(df.head())


    #Create the model, The steps are:
    #1- Create an xception model without the last layer and load the ImageNet pretrained weights
    #2- Add a pre-processing layer
    #3- Add a pooling layer followed by a softmax layer at the end

    #Download the pretrained Xception weights from the S3 Bucket
    download_from_s3('xception_weights_tf_dim_ordering_tf_kernels_notop.h5', model_bucket_name,
                     'xception-pretrained-weights/xception_weights_tf_dim_ordering_tf_kernels_notop.h5')
    
    
    xception_layer = xception.Xception(include_top = False, input_shape = (image_width, image_height,image_channels),
                                       weights = os.path.join(os.getcwd(),  'xception_weights_tf_dim_ordering_tf_kernels_notop.h5'))

    # We don't want to train the imported weights
    xception_layer.trainable = False

    model = Sequential()    
    model.add(keras.Input(shape=(image_width, image_height, image_channels),  name = "inputs"))

    #create a custom layer to apply the preprocessing
    def xception_preprocessing(img):
      return xception.preprocess_input(img)

    model.add(Lambda(xception_preprocessing))
    model.add(xception_layer)
    model.add(tf.keras.layers.GlobalAveragePooling2D())
    model.add(Dense(len(categories), activation='softmax')) 
        
    model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate = learning_rate) , metrics=['categorical_accuracy'])

    model.summary()

    #We will use the EarlyStopping call back to stop our training if the validation_accuray is not improving for a 
    #certain number of epochs.
    early_stop = EarlyStopping(patience = 4, verbose = 1, monitor='val_categorical_accuracy' , mode='max',
                               min_delta=0.0005, restore_best_weights = True)
    callbacks = [early_stop]

    print('call back defined!')


    #Change the categories from numbers to names
    df["category"] = df["category"].replace(categories) 

    
    # The target data set split is: 80% train set, 10% cross_validation set, and 10% test set
    # The test set is already separate. So we take 11.11% of the train-validation set for validation, which corresponds to 10%
    # of the whole data set.
    train_df, validate_df = train_test_split(df, test_size=0.1111, random_state=42)

    train_df = train_df.reset_index(drop=True)
    validate_df = validate_df.reset_index(drop=True)

    total_train = train_df.shape[0]
    total_validate = validate_df.shape[0]

    print('train size = ', total_train , 'validate size = ', total_validate)
    
    #Train the model
    train_datagen = image.ImageDataGenerator()

    train_generator = train_datagen.flow_from_dataframe(
        train_df, 
        train_input_path, 
        x_col='filename',
        y_col='category',
        target_size=(image_width, image_height),
        class_mode='categorical',
        batch_size=batch_size
    )    
    
    #The data set is imbalanced, especially the clothes dataset has much more data than the other categories. So, we 
    #caluclate the weights of the different categories in the training set and then we pass the weight dictionary 
    #as a parameter for the fitting method to take the weights into account
    class_indexes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    class_weights_list = class_weight.compute_class_weight('balanced', np.unique(train_generator.classes), train_generator.classes)
    
    class_weights_dict = dict(zip(class_indexes, class_weights_list))  
    print('class_weights_dict: ', class_weights_dict)
    

    validation_datagen = image.ImageDataGenerator()

    validation_generator = validation_datagen.flow_from_dataframe(
        validate_df, 
        train_input_path, 
        batch_size=batch_size,
        x_col='filename',
        y_col='category',
        class_mode='categorical', 
        target_size=(image_width, image_height)   
    )    

    history = model.fit(
        train_generator, 
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=total_validate//batch_size,
        steps_per_epoch=total_train//batch_size,
        callbacks=callbacks,
        class_weight = class_weights_dict # class weights because the dataset is imbalanced
    )
    

    #Just save a copy of the trained model in s3, just in case!
    model.save("my_model.h5")    
    write_to_s3("my_model.h5", model_bucket_name, "my_model.h5")    
    
    #Save the model
    model.save('/opt/ml/model/1')        
       
    # Plot loss, validation loss, categorical accuracy, and validation categorical accuracy    
    fig, (ax1, ax2) = plt.subplots(2, 1)
    ax1.plot(history.history['loss'], color='b', label="Training loss")
    ax1.plot(history.history['val_loss'], color='r', label="validation loss")
    ax1.set_yticks(np.arange(0, 0.7, 0.1))
    ax1.legend()

    ax2.plot(history.history['categorical_accuracy'], color='b', label="Training accuracy")
    ax2.plot(history.history['val_categorical_accuracy'], color='r',label="Validation accuracy")
    ax2.legend()

    legend = plt.legend(loc='best')
    plt.tight_layout()

    # Save the training curves locally and in s3
    plt.savefig('training_visualization.png', dpi=1200, bbox_inches= 'tight')    
    write_to_s3('training_visualization.png', model_bucket_name, 'training_visualization.png')  