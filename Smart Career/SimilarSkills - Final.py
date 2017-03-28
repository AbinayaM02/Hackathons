# Program to plot the common expertise of two applicants


import xlrd

# Function to read from xlsx file and convert to matrix
def readfromexcel(excelfile):
    workbook = xlrd.open_workbook('C:/Users/amahen/Documents/Expertise_dataset_Transpose.xls')
    worksheet = workbook.sheet_by_name('Dataset for Expertise')
    datamatrix=[]
    i = 0
    
    features = worksheet.col_values(i)[1:]
    for i in range(worksheet.ncols-1):
        row = worksheet.col_values(i+1)
        datamatrix.append(row)
    return datamatrix, features

# Function to compare the similar expertise
def compare(nameone, nametwo, datamatrix):
    employeeone=None
    employeetwo=None
    for rows in datamatrix:
        if rows != []:
            if rows[0] == nameone:
                employeeone = rows
            elif rows[0] == nametwo:
                employeetwo = rows
    indices =[]
    if employeeone and employeetwo:
       for i,j in enumerate(employeeone):
            for k,l in enumerate(employeetwo):
                if j == l and j == 1 and i == k:
                    indices.append(i-1)
                    break
    return indices

# Function to display the common expertise
def display(indices, features, nameone, nametwo):
    skills = []
    print "Common skillset between " + nameone + " and " + nametwo + ":"
    for items in indices:
        #print features[items]
        skills.append(features[items])
    return skills

# Main function
excelfile = 'C:/Users/amahen/Documents/sampledataset.xlsx'
datamatrix, features = readfromexcel(excelfile)
nameone = raw_input('Enter the CEC ID of the first employee:\n')
nametwo = raw_input('Enter the CEC ID of the second employee:\n')
##nameone = 'siarumug'
##nametwo = 'nathangu'
indices = compare(nameone, nametwo, datamatrix)
if indices == None:
    print "One or both of the user names is not found in the database."
else:
    skills = display(indices, features, nameone, nametwo)
    if skills == []:
        print "They don't have any common skills."
    else:
        for items in skills:
            print items
