import torch
import time
import numpy as np

from pytorch_pretrained_bert import BertTokenizer, BertForSequenceClassification
from flask import request

"""
Grammar Mistakes Detector Application 

Detects if the text entered by the user has grammar mistakes or no.

Functions:
----------
run:
	This function starts the application and return the prediction.
"""

# model used to detect grammar errors
model = ''

#tokenizer that converts the enetert text to tokens for the model
tokenizer = ''

# path to the pretrained model
grammar_model_path = 'grammar_model.pt'


# Function that loads the pre-trained model only once
def load_model(grammar_model_path):
	
	global model
	
	if model == '':
		model = torch.load(grammar_model_path)
		model.eval()
	
	return model

	
# Function that loads the tokenizier only once
def load_tokenizer():

	global tokenizer

	if tokenizer == '':
		tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)	
	
	return tokenizer
	
# Function that predicts whether a given sentence has a grammar mistake or no
def predict(model, tokenizier, text_to_analyze):
	
	# maximum length of words that the model can take, if the text_to_analyze is bigger the rest will be truncated, if 
	# it is smaller then it will be padded with zero till it has the MAX_LEN
	MAX_LEN = 64

	sentence = text_to_analyze 
	# The BERT model requires the CLS and SEP labels
	sentence = ["[CLS] " + sentence + " [SEP]"]
	
	#Tokenize the input text
	tokenized_text = [tokenizer.tokenize(sent) for sent in sentence]
	input_ids = [tokenizer.convert_tokens_to_ids(x) for x in tokenized_text]

	# Pad our input tokens
	# The input ids is a two dimensional array, and the first array has only one element
	# We extract the first element, do the padding and return it again.
	input_ids = input_ids[0]

	input_ids_length = len(input_ids)

	# Trim the input ids if it exceeds MAX_LEN
	if (input_ids_length > MAX_LEN):
		input_ids = input_ids [0 : MAX_LEN]

	pad_length = MAX_LEN - len(input_ids)
	
	input_ids = np.pad(input_ids, (0, pad_length), 'constant', constant_values=(0, 0))

	input_ids = [input_ids]

	# Create attention masks
	attention_masks = []

	# Create a mask of 1s for each token followed by 0s for padding
	for seq in input_ids:
	  seq_mask = [float(i>0) for i in seq]
	  attention_masks.append(seq_mask)
	
	# Conversion to torch tensors is a BERT perquisite
	input_ids = torch.tensor(input_ids)
	attention_masks = torch.tensor(attention_masks)
	
	# We don't need to save the gradients while doing the forward pass.
	with torch.no_grad():
	  # Forward pass, calculate logit predictions
	  input_ids = input_ids.type(torch.LongTensor) 
	  logits = model(input_ids, token_type_ids=None, attention_mask=attention_masks)
		
	
	if (logits[0][0] < logits[0][1]):
		return 'no_mistake'
	else:
		return 'mistake'
		
		
def run():

	""" 
	Start the grammar mistakes detection application, read the user input text and return the prediction.

	returns:
		'mistake' if grammar mistake detected, 'no_mistake' otherwise.

	"""

	global model
	global tokenizer
	global grammar_model_path
	
	# text input by the user
	text_to_analyze = ''
	
	# prediction that will be returned
	prediction = ''
	
	model = load_model(grammar_model_path)
	
	tokenizier = load_tokenizer()
	
	# Text that will be shown by default to the user in te text area
	text_to_show = 'Enter text here ...'
	
	# this statement is true if the user gave an input and pressed the Analyze button
	if request.method == 'POST' and 'text_input' in request.form:
		
		# Read the text entered by the user
		text_to_analyze = request.form.get('text_input')
		
		# If the text is the default text, or no text was entered
		if text_to_analyze == '' or text_to_analyze == text_to_show :
			prediction = ''
		else:
		    # This line shows the text eneterd by the user even after the analyzing is done
			text_to_show = text_to_analyze
			prediction = predict(model, tokenizier, text_to_analyze)

	return prediction, text_to_show