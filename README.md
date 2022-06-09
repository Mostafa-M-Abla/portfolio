This Repository contains two Machine Learning Projects, Garbage Classification and Grammar Mistakes Detector, that I implemented and deployed. Feel free to play a bit with the applications! http://mostafa-abla.com/

# Project 1: Garbage Classifier
This project is a demonstration of the ability of machine learning to sort household garbage images into 12 different categories and thus facilitate the recycling process.

As a part of this project the following was done:
-	I collected a data set of more than 15,000 images , saved in the “Datasets folder” and has a “description” file.
-	I wrote a notebook to train the model on classifying the images, the notebook is saved as “model_training_notebook.ipynb” you can also open the notebook in Kaggle using the link https://www.kaggle.com/mostafaabla/garbage-classification-keras-transfer-learning you can also run and edit the notebook there. The notebook has detailed description of what is being done.
-	I wrote a web application that takes an image url and predicts to which class does this image belongs (website mentioned above).

# Project 2: Grammar Mistakes Detector
- This application can detect if the given text has a grammar mistake or no.
- For training the model for this application  the "Corpus of Linguistic Acceptability (CoLA)" data set was used. The data set contains 9594 sentences which are labeled as grammatically correct or no.
- The application was deployed in the website mentioned above, fell free to try it out!

# Repository Contents
1.	"aws_sagemaker_pipeline" Folder containg the complete AWS Sagemaker pipline repos
2.	“DataSets” Folder with description files
3.	“models_training_notebooks” which contain the notebooks used for training both models. 
4.	"garbage_model.json” and "garbage_model_weights.h5” which are the trained model and the weights of the trained garbage classifier model respectively, they are used to perform the prediction.
5.	“static” folder contains some images for the web app.
6.	“template” folder has the html files for the web app
7.	“app.py” is the main file for the web app, and it also uses the “grammar_app.py” and "garbage_app.py" files.
8.	“Dockerfile” is the file from which I created the docker image that I deployed.
9.	Requirements.txt has a list of all the required packages to run the application.

# How to run
-	To test the web app, visit http://mostafa-abla.com/
-	To run the Garbage Classifier model training notebook visit https://www.kaggle.com/mostafaabla/garbage-classification-keras-transfer-learning
-   To run the Grammar Mistakes Detector training notebook, you can run the notebook in Google Colab, saved in models_training_notebooks or also through this link https://colab.research.google.com/drive/1hwxc9p5910KtYHYHIwEk8e3A27PczvDr?usp=sharing
-	If you want to run the web app on your PC then you need to install all the packages mentioned in the “requirements.txt” and then start the “app.py”
-  "grammar_model.pt" which is the trained model for the Grammar mistakes detector app is not included in this repo because it was too big, you need to get it from this link https://drive.google.com/file/d/19HtngA90cFs0XkBGpKvoQGOCICjRMj1q/view?usp=sharing and add the file to this directory