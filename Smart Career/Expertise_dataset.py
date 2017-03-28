from lxml import html
import requests
from bs4 import BeautifulSoup
from urllib import urlopen
from StringIO import StringIO

import xlrd
import xlwt


def getExpertise(id):
    url = 'http://wwwin-tools.cisco.com/dir/details/'+ id
    page = requests.get(url)
    tree = html.fromstring(page.text)
    f = open('workfile', 'w')
    f.write(page.text)
    
    soup = BeautifulSoup(page.text)
    
    tables =  soup.findAll('a')[:-3]
    flag=0
    Expertise=[]
    for table in tables:
        if (table.get_text().strip()):
            if(table.get_text().strip()=='Add your own content'):
                break
            if(flag==2):
                Expertise.append(table.get_text())
                #print(table.get_text())
                #print('\n')
            if(table.get_text().strip()=='Learn more'):
                flag+=1
    return Expertise
    
#def writeDatasetToExcel(Expertise_list,SupersetOfExpertise):
#    workbook=xlwt.Workbook()
#    sheet = workbook.add_sheet("Dataset for Expertise")
#    sheet.write(0, 0, 'Users\\Expertise')
#    i=1
#    for key in SupersetOfExpertise:
#        sheet.write(0,i,key)
#        i+=1
#    i=1
#    for key in Expertise_list:
#        j=0
#        sheet.write(i,j,key)
#        j+=1
#        for Expertise in SupersetOfExpertise:
#            if Expertise in Expertise_list[key]:
#                sheet.write(i,j,1)
#            else:
#                sheet.write(i,j,0)
#            j+=1
#        i+=1
#    
#    workbook.save("Expertise_dataset.xls")  
    
# Transposed dataset is created to overcome column_limit problem in xlwt
def writeDatasetToExcel(Expertise_list,SupersetOfExpertise):
    workbook=xlwt.Workbook()
    sheet = workbook.add_sheet("Dataset for Expertise")
    sheet.write(0, 0, 'Expertise\\Users')
    i=1
    for key in SupersetOfExpertise:
        sheet.write(i,0,key)
        i+=1
    j=1
    for key in Expertise_list:
        i=0
        sheet.write(i,j,key)
        i+=1
        for Expertise in SupersetOfExpertise:
            if Expertise in Expertise_list[key]:
                sheet.write(i,j,1)
            else:
                sheet.write(i,j,0)
            i+=1
        j+=1
    
    workbook.save("Expertise_dataset_Transpose.xls")  

# Need to print user and corresponding Expertise for manual tagging of job profile
def writeUserExpertiseDataToExcel(Expertise_list):
    workbook=xlwt.Workbook()
    sheet = workbook.add_sheet("User and Expertise Data")
    sheet.write(0, 0, 'Expertise\\Users')
    j=1
    for key in Expertise_list:
        i=0
        sheet.write(i,j,key)
        i+=1
        for Expertise in Expertise_list[key]:
            sheet.write(i,j,Expertise)
            i+=1
        j+=1
    
    workbook.save("Expertise_User_Transpose.xls")

def getListOfUsers():
    book = xlrd.open_workbook('ids.xls')
    worksheet = book.sheet_by_index(0)
    num_rows = worksheet.nrows - 1
    num_cells = worksheet.ncols - 1
    curr_row = -1
    UserList=[]
    while curr_row < num_rows:
        curr_row += 1
        #row = worksheet.row(curr_row)
        curr_cell = -1
        while curr_cell < num_cells:
            curr_cell += 1
            # Cell Types: 0=Empty, 1=Text, 2=Number, 3=Date, 4=Boolean, 5=Error, 6=Blank
            #cell_type = worksheet.cell_type(curr_row, curr_cell)
            cell_value = worksheet.cell_value(curr_row, curr_cell)
            UserList.append(cell_value)
    return UserList
    


## Get the Expertise list for a given list of users
#users = ['sakommu','mabigger','liwan3']
users = getListOfUsers()
print len(users)
Expertise_list = {}
for user in users:
    Expertise = getExpertise(user)
    if(len(Expertise)!=0):
        Expertise_list[user] = Expertise

#print Expertise_list

print len(Expertise_list)
## Build superset of Expertise list which becomes the feature list for classification
SupersetOfExpertise=[]
for key in Expertise_list:
    SupersetOfExpertise = list(set(SupersetOfExpertise) | set(Expertise_list[key]))
print "\n\n"
#print SupersetOfExpertise


## Write the dataset into Excel file
writeDatasetToExcel(Expertise_list,SupersetOfExpertise)

writeUserExpertiseDataToExcel(Expertise_list)
