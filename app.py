import os
import time
import tensorflow
import tensorflow.keras.applications.xception as xception

from flask import Flask, request, render_template, redirect

import garbage_app
import grammar_app


''' Flask Application that contains two machine learning applications; Garbage Classifier and Grammar Mistakes Detector'''


app = Flask(__name__)

	
@app.route('/', methods = ['GET', 'POST'])
def rootpage():
	'''Home page of the website '''

	return render_template("index.html")

	
@app.route('/garbage', methods = ['GET', 'POST'])
def garbage():
	'''Page of the garbage classifier app '''

	image_to_classify_name, pred_visualisation_name, category_1_name, category_1_prob, category_2_name, category_2_prob, category_3_name,\
	  category_3_prob = garbage_app.run()
		

	return render_template("garbage.html", image_to_classify_name = image_to_classify_name, pred_visualisation_name = pred_visualisation_name,
							category_1_name = category_1_name, category_1_prob = category_1_prob,
							category_2_name = category_2_name, category_2_prob = category_2_prob,
							category_3_name = category_3_name, category_3_prob = category_3_prob)
							
							
@app.route('/grammar', methods = ['GET', 'POST'])
def grammar():
	'''Page of the Grammar Mistakes Detector app '''
	
	prediction, text_to_show = grammar_app.run()
	
	return render_template("grammar.html", prediction = prediction, text_to_show = text_to_show)
				
				
app.run(host='0.0.0.0', port=5000)