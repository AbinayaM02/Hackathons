# -*- coding: utf-8 -*-
"""
Created on Sat Sep 22 14:31:03 2018

@author: Abinaya.M02
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Sep 22 11:10:58 2018

@author: Abinaya.M02
"""


# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
import pickle

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
                         ('clf', SGDClassifier(loss='hinge', penalty='l2', alpha=1e-4, random_state=24,max_iter=5, tol=None)),
                     ])
    return(text_clf)

# Main
if __name__ == '__main__':
  
    # Read the data and take the necessary columns and rows
    filename = "../Data/query_train_clean.csv"
    data_train = pd.read_csv(filename,  keep_default_na = False)
    data_train = data_train[data_train['Query'] != ""]
    print(data_train['Label'].value_counts())
          
    # Split the data
    X_train, y_train, X_test, y_test = split_data(data_train)
    print("Data Split Measure: ")
    print("X_train Shape: ", X_train.shape)
    print("y_train Shape: ", y_train.shape)
    print("X_test Shape: ", X_test.shape)
    print("y_test Shape: ", y_test.shape)
    
    # Parameters for the model
    parameters = {
    'vect__max_df': (0.5, 0.75, 1.0),
    'vect__max_features': (None, 5000, 10000, 50000),
    'vect__ngram_range': ((1, 1), (1, 2)),  # unigrams or bigrams
    'tfidf__use_idf': (True, False),
    'tfidf__norm': ('l1', 'l2'),
    'clf__alpha': (0.00001, 0.000001),
    'clf__penalty': ('l2', 'elasticnet'),
    'clf__n_iter': (10, 50, 80),
    }

    grid_search = GridSearchCV(pipeline(), parameters, n_jobs=-1, verbose=1)

    print("Performing grid search...")
    print("pipeline:", [name for name, _ in pipeline().steps])
    print("parameters:")
    grid_search.fit(X_train, y_train)


    print("Best score: %0.3f" % grid_search.best_score_)
    print("Best parameters set:")
    best_parameters = grid_search.best_estimator_.get_params()
    for param_name in sorted(parameters.keys()):
        print("\t%s: %r" % (param_name, best_parameters[param_name]))

#    # Train the model
#    text_clf = pipeline().fit(X_train, y_train)
#    
#    # Save the model
#    pickle.dump(text_clf, open("../Model/svm.pkl", 'wb'))
#   
#    # Get the parameters of the estimators
#    print(text_clf.get_params)
#    
#    # Predict the new data points
#    print('Predicting...')
#    #predicted = text_clf.predict(X_test)
#    loaded_model = pickle.load(open("../Model/svm.pkl", 'rb'))
#    predicted = loaded_model.predict(X_test)
#    print(predicted)
#    
#    # Measure the accuracy
#    print('Accuracy Score: ',metrics.accuracy_score(y_test, predicted))
#    print(metrics.classification_report(y_test, predicted))
#    
