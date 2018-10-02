# -*- coding: utf-8 -*-
"""
Script for prediction with different models

Created on Sat Sep 22 13:27:12 2018

@author: Abinaya.M02
"""

# Import necessary libraries
import pandas as pd
from keras.models import load_model
import pickle 
from keras.preprocessing import sequence
import numpy as np


# Main
if __name__ == '__main__':
    data_test = pd.read_csv("../Data/query_test_clean.csv")
    
#    # Load the saved model and labels
#    loaded_model_nb = pickle.load(open("../Model/NB.pkl", 'rb'))
#    loaded_model_sgd = pickle.load(open("../Model/svm.pkl", 'rb'))
#    loaded_model_rf = pickle.load(open("../Model/rf.pkl", 'rb'))
   
    # Load the saved model and labels
    loaded_model_w2v = load_model('../Model/lstm_model_10.h5')
    
    tokenizer = pickle.load(open('../Model/tokenizer_10.pkl', 'rb'))
    
    # Predict the category for the search results' title
    max_len = 150
    req_data_title = tokenizer.texts_to_sequences(data_test['Query'])
    req_data_title = sequence.pad_sequences(req_data_title, max_len)
    print(req_data_title.shape)
    
    # Predcit the category
    data_test['Label'] = [np.argmax(entry) for entry in loaded_model_w2v.predict(req_data_title)]
    data_test = data_test[['QueryId', 'Label']]
    data_test.to_csv("../Data/word2vec_test.csv", index = False)