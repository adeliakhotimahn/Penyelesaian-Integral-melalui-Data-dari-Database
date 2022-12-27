import numpy as np
import csv
import time

from sklearn import svm
import pandas as pd

#Database: Gerbang LOgika AND
#Membaca data dari file
FileDB = 'database.txt'
Database = pd.read_csv(FileDB, sep=",", header=0)
print(Database)



#x = data, y = target
x = Database[[u'a',u'b',u't']]
y = Database.t

clf = svm.SVC()
clf.fit(x,y)

print(clf.predict( [[2,5]] ))
print(clf.predict( [[4,6]] ))
print(clf.predict( [[5,7]] ))
print(clf.predict( [[3,5]] ))
