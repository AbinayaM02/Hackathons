# Program to plot the common expertise of two applicants


import xlrd
import operator
from sklearn import svm
from sklearn.ensemble import ExtraTreesClassifier
import numpy as np

# Function to read from xlsx file and convert to matrix
def readfromexcel(excelfile):
    workbook = xlrd.open_workbook(excelfile)
    worksheet = workbook.sheet_by_name('Dataset for Expertise')
    datamatrix=[]
    i = 0
    
    features = worksheet.col_values(i)[1:worksheet.nrows-1]
    for i in range(worksheet.ncols-1):
        row = worksheet.col_values(i+1)[1:worksheet.nrows-1]
        datamatrix.append(row)
    Y = worksheet.row_values(worksheet.nrows-1)[1:worksheet.nrows-1]
    return datamatrix, Y, features

def ranking(X,Y,features):
    # Build a forest and compute the feature importances
    forest = ExtraTreesClassifier(n_estimators=250,random_state=0)
    forest.fit(X, Y)
    importances = forest.feature_importances_std = np.std([tree.feature_importances_ for tree in forest.estimators_],axis=0)
    indices = np.argsort(importances)[::-1]

    # Print the feature ranking
    featureDict={}
    for f in range(10):
        if str(features[indices[f]]).lower() in featureDict:
            featureDict[str(features[indices[f]]).lower()] +=importances[indices[f]]
        else:
            featureDict[str(features[indices[f]]).lower()] =importances[indices[f]]
        if len(featureDict)==5:
            break
    sorted_features = sorted(featureDict.items(), key=operator.itemgetter(1),reverse=True)
    i = 1
    for feature,importance in sorted_features:
        print("%d. %s: (%f)" % (i, feature, importance))
        i += 1
    print "\n"
    
# Main function
profiles = ["Manager", "Software Engineering - Dev","Software Engineering - QA","Data Engineer","IT Engineer","Sales","Network Consultant"]

for i in range(7):
    print("Feature ranking for the profile - "+profiles[i] +":")
    excelfile = 'C:/Users/amahen/Documents/Expertise_classWise/Expertise_dataset_Transpose_class' + str(i+1) + '.xls'
    X, Y, features = readfromexcel(excelfile)
    ranking(X,Y,features)

    

