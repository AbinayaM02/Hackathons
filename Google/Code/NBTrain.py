# -*- coding: utf-8 -*-
"""
Created on Sat Sep 22 10:38:04 2018

@author: Abinaya.M02
"""

# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn import metrics
import pickle
import re

# Clean the text  
def remove_junk(data):
    #stopwords = ['-','.','”','“','*','!']
    lines = [re.sub('[^A-Za-z0-9"\ "$]+', '', line) for line in data if line.strip()]
    return(lines)
    
    
# Function to split the data into train and test
def split_data(data):
    """ Split the data into train and test. Return X_train, y_train, X_test and y_test and the labels"""
    X_train, X_test, y_train, y_test = train_test_split(data['Query'], data['Label'], test_size = 0.33, random_state = 42)
    return(X_train, y_train, X_test, y_test)

# Function to setup the pipeline for training
def pipeline():
    """ Setup the pipleline for training and return the object"""
    print('Training...')

    text_clf = Pipeline([('vect', CountVectorizer(min_df=1e-6, max_df=0.1, stop_words='english', strip_accents = 'unicode', analyzer = 'word')),
                         ('tfidf', TfidfTransformer(use_idf=False)),
                         ('clf', MultinomialNB()),
                     ])
    return(text_clf)

# Main
if __name__ == '__main__':
  
    # Read the data and take the necessary columns and rows
    filename = "../Data/query_train_clean.csv"
    data_train = pd.read_csv(filename,  keep_default_na = False)
    data_train = data_train[data_train['Query'] != ""]
    #data_train['Query'] = remove_junk(data_train['Query'])
    #print(data_train['Label'].value_counts())
          
    # Split the data
    X_train, y_train, X_test, y_test = split_data(data_train)
    print("Data Split Measure: ")
    print("X_train Shape: ", X_train.shape)
    print("y_train Shape: ", y_train.shape)
    print("X_test Shape: ", X_test.shape)
    print("y_test Shape: ", y_test.shape)
    
    # Train the model
    text_clf = pipeline().fit(X_train, y_train)
    
    # Save the model
    pickle.dump(text_clf, open("../Model/nb.pkl", 'wb'))
   
    # Get the parameters of the estimators
    print(text_clf.get_params)
    
    # Predict the new data points
    print('Predicting...')
    #predicted = text_clf.predict(X_test)
    loaded_model = pickle.load(open("../Model/nb.pkl", 'rb'))
    predicted = loaded_model.predict(X_test)
    print(predicted)
    
    # Measure the accuracy
    print('Accuracy Score: ',metrics.accuracy_score(y_test, predicted))
    print(metrics.classification_report(y_test, predicted))
    
