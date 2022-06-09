import numpy as np
import time
import platform
import os
import requests
import urllib

from flask import request

"""
Garbage Classifier Application 

This application classifies household garbage images into 12 categories.

"""
# Each time the user selects an image to classify we first delete the old saved images
def deleteOldImages():
	
	for filename in os.listdir('static'):
		if filename.startswith('image_to_predict'):  # not to remove other images
			os.remove('static/' + filename)
			
	return

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

	#Address where the post request is sent
	api_gateway = 'https://ne2tdxki81.execute-api.eu-central-1.amazonaws.com/predict'
    
	# Before the user selects an image to classify, place a blank image just as a placeholder.
	image_to_classify_name = 'static/white_background.jpg'
	
	# The image showing the 12 possible garbage categories
	pred_visualisation_name = 'static/garbage_categories/default.jpg'
		
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
			#load and save the image from the url
			urllib.request.urlretrieve(url, image_to_classify_name)

			#Send the post request to the API-Gateway
			r = requests.post(api_gateway, json={"url": url})

			response = r.json()

			category_1_name = response['category_1_name']
			category_1_prob = response['category_1_probability']
			category_2_name = response['category_2_name']
			category_2_prob = response['category_2_probability']
			category_3_name = response['category_3_name']
			category_3_prob = response['category_3_probability']
            

			# After we predict that the image belongs to a certain category, we show an image of that category selected
			pred_visualisation_name = 'static/garbage_categories/' +  category_1_name + '.jpg'
			
		# If the user entered an invalid url display an error message
		except Exception as err:
			print("err: ", err)	      
			image_to_classify_name = 'static/Warning.jpg'

	return image_to_classify_name, pred_visualisation_name,	category_1_name, category_1_prob, \
			category_2_name, category_2_prob, category_3_name, category_3_prob							