# Program to plot the common expertise of two applicants


import xlrd
from sklearn import svm
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

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
    X = datamatrix
    Y = worksheet.row_values(worksheet.nrows-1)[1:worksheet.nrows-1]

    return X, Y, features

    
# Main function
excelfile = 'C:/Users/amahen/Documents/Expertise_dataset_Transpose.xls'
X, Y, features = readfromexcel(excelfile)

# Training
lin_clf = svm.LinearSVC()
Xtrain = X[:180]
Ytrain = Y[:180]
lin_clf.fit(Xtrain, Ytrain)

# Label for the class
print " Class                     :       Label \n"
print "Manager                    :       1\n"    
print "Software Engineering - Dev :       2\n"    
print "Software Engineering - QA  :       3\n"    
print "Data Engineer              :       4\n"    
print "IT Engineer                :       5\n"    
print "Sales                      :       6\n"    
print "Network Consultant         :       7\n"

# Testing
Xtest = X[180:]
Ytest = Y[180:]
Ypred = lin_clf.predict(Xtest)
#print Ypred, Ytest


### Split the data into a training set and a test set
##X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=150)
##
### Run classifier
##classifier = svm.SVC(kernel='linear')
##y_pred = classifier.fit(X_train, y_train).predict(X_test)

# Compute confusion matrix
cm = confusion_matrix(Ytest, Ypred)

print(cm)

# Show confusion matrix in a separate window
plt.matshow(cm)
plt.title('Confusion matrix')
plt.colorbar()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()

##print features, X , Y
##nameone = 'svs'
##nametwo = 'nwhite'
##indices = compare(nameone, nametwo, datamatrix)
##skills = display(indices, features, nameone, nametwo)
