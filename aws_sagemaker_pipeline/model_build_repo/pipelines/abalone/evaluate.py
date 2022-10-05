"""Evaluation script for measuring F1 Score macro average"""
import json
import logging
import pathlib
import pickle
import tarfile

import numpy as np
import pandas as pd
import os
import sys
import math
import re
import tensorflow.keras as keras
import tensorflow as tf
import tensorflow.keras.applications.xception as xception
import tensorflow.python.keras.applications.xception as xception
import boto3
import argparse

from sagemaker.tensorflow import TensorFlow
from tensorflow.keras.preprocessing import image
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model
from PIL import Image
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sagemaker.debugger import Rule, rule_configs


logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())



if __name__ == "__main__":
    
    print("Starting evaluation ...")
    
    def write_to_s3(filename, bucket, key):
        with open(filename,'rb') as f: # Read in binary mode
            return boto3.Session().resource('s3').Bucket(bucket).Object(key).upload_fileobj(f)
        
    # Add class name prefix to filename. So for example "/paper104.jpg" become "paper/paper104.jpg"
    def add_class_name_prefix(df, col_name):
        df[col_name] = df[col_name].apply(lambda x: x[:re.search("\d",x).start()] + '/' + x)
        return df       
    
    # Method to load an image and resize it into the required size
    def load_image(path):
        img = image.load_img(path, target_size = (image_width, image_height)) # load the image from the directory
        img = image.img_to_array(img) 
        # add an additional dimension, (i.e. change the shape of each image from (224, 224, 3) to (1, 224, 224, 3)
        # This shape is suitable for training
        img = np.expand_dims(img, axis = 0)         
        return img
    
    #Load the trained model and unzip it.
    model_path = "/opt/ml/processing/model/model.tar.gz"
    with tarfile.open(model_path) as tar:
        def is_within_directory(directory, target):
            
            abs_directory = os.path.abspath(directory)
            abs_target = os.path.abspath(target)
        
            prefix = os.path.commonprefix([abs_directory, abs_target])
            
            return prefix == abs_directory
        
        def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
        
            for member in tar.getmembers():
                member_path = os.path.join(path, member.name)
                if not is_within_directory(path, member_path):
                    raise Exception("Attempted Path Traversal in Tar File")
        
            tar.extractall(path, members, numeric_owner=numeric_owner) 
            
        
        safe_extract(tar, path="/tmp/")
        
    # Parse the passed parameters
    parser = argparse.ArgumentParser()     
    parser.add_argument('--image_width', type=str, default = "320")
    parser.add_argument('--image_height', type=str, default = "320")
    parser.add_argument('--image_channels', type=str, default = "3")

    args, _ = parser.parse_known_args()
    image_width = int(args.image_width)
    image_height = int(args.image_height)
    image_channels = int(args.image_channels)

    print('Parameters passed: ')
    print('args.image_width : ', image_width)
    print('args.image_height : ', image_height)
    print('args.image_channels : ', image_channels)
    
    # Dictionary to save our 12 classes
    categories = {0: 'battery', 1: 'biological', 2: 'brown-glass', 3: 'cardboard', 4: 'clothes', 5: 'green-glass',
                  6: 'metal', 7: 'paper', 8: 'plastic', 9: 'shoes', 10: 'trash', 11: 'white-glass'}
    
    #Test Images are saved here
    test_input_path="/opt/ml/processing/test"     
                
    model = load_model('/tmp/1')    
    print("model.summary()", model.summary())            
    
    #We create a dataframe, that have the list of names of our test images and the corresponding categories
    
    # list conatining all the filenames in the test dataset
    filenames_list = []

    # list to store the corresponding category, note that each folder of the dataset has one class of data
    categories_list = []

    for category in categories:
        filenames = os.listdir(test_input_path + '/' + categories[category])

        filenames_list = filenames_list  + filenames
        categories_list = categories_list + [category] * len(filenames)

    df = pd.DataFrame({
        'filename': filenames_list,
        'category': categories_list
    })

    df = add_class_name_prefix(df, 'filename')

    # Shuffle the dataframe
    df = df.sample(frac=1).reset_index(drop=True)

    #Change the categories from numbers to names
    df["category"] = df["category"].replace(categories) 
    
    print("evaluation dataframe length = ", len(df))
    
    
    ## Evaluate the Test data set
    
    test_set = df.to_numpy()

    predictions_complete = []
    labels_complete=[]

    test_set_size = np.shape(test_set)[0]
    print('test_set_size:' , test_set_size)

    #We will predict 80 images at a time, becuase we need to load them and save them in the df.
    # We simply don't want to consume too much RAM
    step_size = 80
    number_of_iterations = math.ceil(test_set_size / step_size)

    for iteration in range (number_of_iterations):

        start_index = iteration*step_size
        if (iteration < number_of_iterations -1):        
            test_subset = test_set[start_index : start_index + step_size, : ]
        else:
            test_subset = test_set[start_index : test_set_size, : ]

        #Create empty input data and labels arrays
        test_x = np.zeros((len(test_subset), image_width, image_height, image_channels )) 
        test_y = np.zeros((len(test_subset) ))

        # prepare train_x and train_y
        for i, test_entry in enumerate (test_subset):
            file_name = test_entry[0]
            category  = test_entry[1]   

            test_y[i] = [k for k, v in categories.items() if v == category][0]  
            path = test_input_path + '/' + file_name  
            loaded_image = load_image(path)

            test_x[i, :, :, :] = loaded_image    

        test_y = to_categorical(test_y)     

        #This is the real prediction step
        predictions = model.predict(test_x)

        #Get the class with the highest preodiction probability and compare it with the label
        predictions = np.argmax(predictions, axis = 1)

        predictions = [categories[item] for item in predictions]

        labels = np.argmax(test_y, axis = 1)

        labels = [categories[item] for item in labels]

        predictions_complete = predictions_complete + predictions
        labels_complete = labels_complete + labels

    print(classification_report(labels_complete, predictions_complete))

    # Calculate the macro_avg of the F1 score all classes, this is not the weighted average.
    # We parse the macro_ag from the F1 score report
    report = classification_report(labels_complete, predictions_complete)

    weighted_averageLine = report.splitlines()[-2]
    macro_avg = float(weighted_averageLine.split()[4])
    print("macro_avg = ", macro_avg)    
        
    report_dict = {
        "multiclass_classification_metrics": {
            "accuracy": {
                "value": macro_avg,
                "standard_deviation": "NaN"
            },
        },
    }    
    
    output_dir = "/opt/ml/processing/evaluation"
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    evaluation_path = f"{output_dir}/evaluation.json"
    with open(evaluation_path, "w") as f:
        f.write(json.dumps(report_dict))
        
    print("evalaute.py end")