# -*- coding: utf-8 -*-
"""
Created on Sat Sep 22 14:15:40 2018

@author: Abinaya.M02
"""

# Import necessary libraries
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import pickle

# Fucntion to perform one hot encoding
def one_hot_encoding(data):
    """ Convert the categorical data to one hot encoding. Return the one hot encoded values"""
    encoder = LabelEncoder()
    y = encoder.fit_transform(data)
    return(y)


# Function to split the data into train and test
# Function to split the data into train and test
def split_data(data):
    """ Split the data into train and test. Return X_train, y_train, X_test and y_test"""
    # Split the data into train and test
    X_train, X_test, y_train, y_test = train_test_split(data['Query'], data['Label'], test_size=0.30, random_state = 42)
    return(X_train, y_train, X_test, y_test)

# Function to setup the pipeline for training
def pipeline():
    """ Setup the pipleline for training and return the object"""
    print('Training...')

    text_clf = Pipeline([('vect', CountVectorizer(min_df=1e-6, max_df=0.1, stop_words='english')),
                     ('tfidf', TfidfTransformer(use_idf=False)),
                     ('clf', RandomForestClassifier()),
                     ])
    return(text_clf)

# Main
if __name__ == '__main__':
#def main():   
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
    pickle.dump(text_clf, open("../Model/rf.pkl", 'wb'))
    
    # Predict the new data points
    print('Predicting...')
    #predicted = text_clf.predict(X_test)
    loaded_model = pickle.load(open("../Model/rf.pkl", 'rb'))
    predicted = loaded_model.predict(X_test)
    print(predicted)
    
    # Measure the accuracy
    print('Accuracy Score: ',metrics.accuracy_score(y_test, predicted))
    print(metrics.classification_report(y_test, predicted))
    