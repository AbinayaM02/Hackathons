# -*- coding: utf-8 -*-
"""
Created on Sat Sep 22 11:38:30 2018

@author: Abinaya.M02
"""


# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from sklearn.preprocessing import LabelEncoder
from keras.layers import LSTM, Dense, Embedding, SpatialDropout1D, Conv1D, MaxPooling1D
from keras.utils import to_categorical
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.callbacks import EarlyStopping
import pickle

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
def RNN(emb_dim, max_words, data):
    ''' Define the RNN architecture with LSTM units and return the model'''
    print(data.shape)
    model = Sequential()
    model.add(Embedding(max_words, emb_dim, input_length=data.shape[1]))
    model.add(SpatialDropout1D(0.5))
    #model.add(Conv1D(64, 5, activation='relu'))
    #model.add(MaxPooling1D(pool_size=4))
    model.add(LSTM(64, dropout=0.5, recurrent_dropout=0.5))
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
    sequences_matrix = sequence.pad_sequences(sequences,maxlen=max_len, padding = 'post')
    return(sequences_matrix, tok)
    
# Main
if __name__ == '__main__':
#def main():   
    # Read the data and take the necessary columns and rows
    filename = "../Data/query_train_clean.csv"
    data_train = pd.read_csv(filename, keep_default_na = False)
    data_train = data_train[data_train['Query'] != ""]
   
    # Perform one hot encoding
    ncategories = one_hot_encoding(data_train['Label'])
    y_labels = to_categorical(ncategories)
    
    # Parameter for the sequences
    max_words = 5000
    max_len = 150
    
    # Split the data
    sequences_matrix, tokenizer = tokenize(max_words, max_len, data_train['Query'])
    #print(len(data_train['Query']), len(y_labels), sequences_matrix.shape)
    X_train, y_train, X_test, y_test = split_data(sequences_matrix, y_labels)
    print("Data Split Measure: ")
    print("X_train Shape: ", X_train.shape)
    print("y_train Shape: ", y_train.shape)
    print("X_test Shape: ", X_test.shape)
    print("y_test Shape: ", y_test.shape)
    
    # Define the parameters for RNN 
    emb_dim = 150
    epochs = 10 
    batch_size = 64
    
    # Compile the model
    model = RNN(emb_dim, max_words, sequences_matrix)
    
    # Train the model 
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size,
        validation_split=0.2,callbacks=[EarlyStopping(monitor='val_loss',patience=10, min_delta=0.0001)])
    
    # Test the model
    accr = model.evaluate(X_test,y_test)
    print('Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(accr[0],accr[1]))
    
    # Save the tokenizer
    pickle.dump(tokenizer, open('tokenizer_16.pkl', 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
    
    # Save the model
    model.save('lstm_model_16.h5')