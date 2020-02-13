import csv
import pandas as pd


df = pd.read_csv('ult-good_enroll2.csv')
saved_column = df['correct'] #you can also use df['column_name']
error = 0

for i in range(323):
    if(saved_column[i] == 0):
        error +=1

print(error)
print(error/323)
