# -*- coding: utf-8 -*-
"""
Script to perform text classification using LSTM Word2Vec

Created on Sat Sep 22 14:48:26 2018
@author: Abinaya.M02
"""

# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers import LSTM, Dense, Embedding, SpatialDropout1D
from keras.utils import to_categorical
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.callbacks import EarlyStopping
import pickle
import numpy as np


# Fucntion to perform one hot encoding
def one_hot_encoding(data):
    """ Convert the categorical data to one hot encoding. Return the one hot encoded values"""
    encoder = LabelEncoder()
    y = encoder.fit_transform(data)
    return(y)


# Function to split the data into train and test
def split_data(data, labels):
    """ Split the data into train and test. Return X_train, y_train, X_test and y_test"""
    # Split the data into train and test
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.30, random_state = 42)
    return(X_train, y_train, X_test, y_test)

# Define the RNN architecture
def RNN(max_words, emb_dim, data, max_len, word_index, trainable):
    ''' Define the RNN architecture with LSTM units and return the model'''
    model = Sequential()#word_index, max_len, emb_dim)
    model.add(word_embeddings(word_index, emb_dim, max_len))#Embedding(max_words, emb_dim, input_length=data.shape[1]))
    model.add(SpatialDropout1D(0.7))
    model.add(LSTM(64, dropout=0.7, recurrent_dropout=0.7))
    model.add(Dense(10, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
    print(model.summary())
    return(model)

# Function to tokenize and convert data to sequences
def tokenize(max_words, max_len, data):
    ''' Tokenize the text and convert it into sqeuence and return the sequence'''
    tok = Tokenizer(num_words=max_words, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower = True)
    tok.fit_on_texts(data)
    sequences = tok.texts_to_sequences(data)
    sequences_matrix = sequence.pad_sequences(sequences,maxlen=max_len)
    return(sequences_matrix, tok)
    
# Function to read the word vectors
def word_vectors():
    ''' Return the word embeddings from word2vec'''
    embeddings_index = dict()    
    file = open('../Model/Word2VecFile.txt', encoding="utf-8")    
    for line in file:      
        values = line.split()    
        word = values[0]    
        coefs = np.asarray(values[1:], dtype='float32')    
        embeddings_index[word] = coefs    
    file.close()
    return embeddings_index
    
# Function to create word embeddings for the words and return the embedding_layer
def word_embeddings(word_index, embedding_dim, max_len):
    ''' Generate the word vectors for the embedding layers '''
    embeddings_index = word_vectors()
    embedding_matrix = np.random.random((len(word_index) + 1, embedding_dim))
    for word, i in word_index.items():    
        embedding_vector = embeddings_index.get(word)    
        if embedding_vector is not None:    
            # words not found in embedding index will be all-zeros.    
            embedding_matrix[i] = embedding_vector    
    embedding_layer = Embedding(len(word_index) + 1,    
                                embedding_dim,  
                                weights=[embedding_matrix],  
                                input_length=max_len,  
                                trainable=False)
    return(embedding_layer)
  
    
# Main
if __name__ == '__main__':
#def main():   
    # Read the data and take the necessary columns and rows
    filename = "../Data/query_train_clean.csv"
    data_train = pd.read_csv(filename,  keep_default_na = False)
    data_train = data_train[data_train['Query'] != ""]
    
    # Perform one hot encoding
    ncategories = one_hot_encoding(data_train['Label'])
    y_labels = to_categorical(ncategories)
    #y_labels = data_train['Risk Tag']
       
    
    # Parameter for the sequences
    max_words = 5000
    max_len = 150
    
    # Split the data
    sequences_matrix, tokenizer = tokenize(max_words, max_len, data_train['Query'])
    X_train, y_train, X_test, y_test = split_data(sequences_matrix, y_labels)
    print("Data Split Measure: ")
    print("X_train Shape: ", X_train.shape)
    print("y_train Shape: ", y_train.shape)
    print("X_test Shape: ", X_test.shape)
    print("y_test Shape: ", y_test.shape)
    
    # Define the parameters for RNN 
    emb_dim = 50
    epochs = 10 #10 for full data
    batch_size = 256
    
    # Build the word index
    word_index = tokenizer.word_index
    
    # Compile the model
    model = RNN(max_words, emb_dim, sequences_matrix, max_len, word_index, trainable = False)
    
    # Train the model 
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size,
        validation_split=0.2,callbacks=[EarlyStopping(monitor='val_loss',patience=10, min_delta=0.0001)])
    
    # Test the model
    accr = model.evaluate(X_test,y_test)
    print('Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(accr[0],accr[1]))
    
    # Save the tokenizer
    pickle.dump(tokenizer, open('../Model/tokenizer_10.pkl', 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
    
    # Save the model
    model.save('../Model/lstm_model_10.h5')