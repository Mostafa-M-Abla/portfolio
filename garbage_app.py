import numpy as np
import pandas as pd
import time
import platform
import os
import tensorflow
import urllib

from tensorflow.keras.models import model_from_json
from tensorflow.keras.preprocessing import image
from flask import request

"""
Garbage Classifier Application 

This application classifies household garbage images into 12 categories.

Functions:
----------
run:
	This function starts the application and return the prediction.
"""

# image size in pixels, when images are loaded this size will be used
IMAGE_WIDTH = 320    
IMAGE_HEIGHT = 320

# List of our 12 categories
categories_list = ['battery', 'biological', 'brown-glass', 'cardboard', 'clothes', 'green-glass',
                      'metal', 'paper', 'plastic', 'shoes', 'trash', 'white-glass']

#Paths where our pre-trained model and it's weights are saved
model_h5_path = 'garbage_model_weights.h5'
model_json_path = 'garbage_model.json'

# model that will be used for classification
model = ''


# Each time the user selects an image to classify we first delete the old saved images
def deleteOldImages():
	
	for filename in os.listdir('static'):
		if filename.startswith('image_to_predict'):  # not to remove other images
			os.remove('static/' + filename)
			
	return
	
#Function to load an image and convert it to an array with the right shape
def load_image(img_path):
    img = image.load_img(img_path, target_size = (IMAGE_WIDTH, IMAGE_HEIGHT)) # load the image from the directory
    img = image.img_to_array(img) 
    # add an additional dimension, (i.e. change the shape of each image from (320, 320, 3) to (1, 320, 320, 3)
	# so that the image size is compatible with our model
    img = np.expand_dims(img, axis = 0)    
	
    return img

	
#Load the trained model from the h5 and the json files, the model is only loaded once
def load_model(model_h5_path, model_json_path):
	
	global model
	
	if model == '':		
		# load json and create model
		json_file = open(model_json_path, 'r')

		model_json = json_file.read()

		json_file.close()

		model = model_from_json(model_json)

		# load weights into new model
		model.load_weights(model_h5_path)
	
	return model
	
	
# Predict to which category  the selected image belongs and return the top 3 categories with their probabilities
def predict(model, image_to_classify_name, model_h5_path, model_json_path):

	# return the probabilities of all 12 categories
	pred = model.predict(load_image(image_to_classify_name))

	preds_df = pd.DataFrame({
		'categories': categories_list,
		'preds': pred[0]
	})
	
	# sort the categories with ascending values of the probabilities
	preds_df.sort_values(by=['preds'], inplace=True, ascending=False)

	# convert the probabilities to percentages
	category_1_name = preds_df.categories.iloc[0]
	category_1_prob = round((preds_df.preds.iloc[0]* 100), 1)  
	
	category_2_name = preds_df.categories.iloc[1]
	category_2_prob = round((preds_df.preds.iloc[1]* 100), 1)  
	
	category_3_name = preds_df.categories.iloc[2]
	category_3_prob = round((preds_df.preds.iloc[2]* 100), 1)  
	
	return category_1_name, category_1_prob, category_2_name, category_2_prob, category_3_name, category_3_prob 

	
def run():

	""" 
	Start the Garbage Classifier application, load the image selected by the user,
	and predict the right class.

	return
	--------
	image_to_classify_name:  The image selected by the user is saved and given a name, this is the name.
    pred_visualisation_name: Each category of garbage has a saved image that detects that this class is selected, 
	                         this the name of the image to be shown.
	category_1_name:         Name of the category with the highest probability
    category_1_prob:         Probability of category no. 1
	category_1_name:         Name of the category with the second highest probability
    category_1_prob:         Probability of category no. 2
	category_1_name:         Name of the category with the third highest probability
    category_1_prob:         Probability of category no. 3
	"""
	
	# Each time the user selects an image to classify we first delete the old saved images, before saving the new one.
	deleteOldImages()
	
	# url of the target image to classify
	url = ''
	
	# Before the user selects an image to classify, place a blank image just as a placeholder.
	image_to_classify_name = 'static/white_background.jpg'
	
	# The image showing the 12 possible garbage categories
	pred_visualisation_name = 'static/garbage_categories/default.jpg'
	
	model = load_model(model_h5_path, model_json_path)
	
	# The three top predictions and their respective probabilities
	category_1_name = ''
	category_1_prob = ''
	category_2_name = ''
	category_2_prob = ''
	category_3_name = ''
	category_3_prob = ''
	
	# this statement is true if the user gave an input and pressed the classify button
	if request.method == 'POST' and 'url-input' in request.form:
		
		# Read the url given by the user
		url = request.form.get('url-input')
		request_time = time.time()
		image_to_classify_name =  "static/image_to_predict" +str(request_time)+ ".jpg"
		
		# Save the image selected by the user and predict the top 3 categories
		try:
			urllib.request.urlretrieve(url, image_to_classify_name)		
			category_1_name, category_1_prob, category_2_name, category_2_prob, category_3_name, category_3_prob = \
				predict(model, image_to_classify_name, model_h5_path, model_json_path)
			
			# After we predict that the image belongs to a certain category, we show an image of that category selected
			pred_visualisation_name = 'static/garbage_categories/' +  category_1_name + '.jpg'
			
		# If the user entered an invalid url display an error message
		except Exception:	      
			image_to_classify_name = 'static/Warning.jpg'

	return image_to_classify_name, pred_visualisation_name,	category_1_name, category_1_prob, \
			category_2_name, category_2_prob, category_3_name, category_3_prob							