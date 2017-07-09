#Example: python dataPreProcessing.py "FiveFoldData" "workdata.csv" 

import sys
import os
import numpy as np
from sklearn.preprocessing import Imputer
import warnings
warnings.filterwarnings('ignore')

if len(sys.argv)<2:
  print "Wrong No: of Input Parameters"
  print "Required format:"
  print "Argument 1: Folder to store the Five Fold data(Will automatically create the folder if it doesn't exist)"
  print "Argument 2: Name of the data file"
  
NewDataFolder = sys.argv[1]
DataFile = sys.argv[2]
if not os.path.exists(NewDataFolder):
  os.makedirs(NewDataFolder)

raw_data = open(DataFile)
dataset = np.loadtxt(raw_data, delimiter=",")

#Using sklearn imputer(mean), you can change this function and replace with appropriate imputer(Option K-Means)
def mean_X_Filler(X):
  imp = Imputer(missing_values=0.0, strategy='mean', axis=0)
  return imp.fit_transform(X)
  
def saveData(test, rest, fold):
  threeFouth = int(.75*len(rest))
  train = rest[:threeFouth,:]
  validation = rest[threeFouth:,:]
  if not os.path.exists(NewDataFolder+"/"+fold):
    os.makedirs(NewDataFolder+"/"+fold)
    
  np.savetxt(NewDataFolder+"/"+fold+"/train.txt", train, fmt='%f')
  np.savetxt(NewDataFolder+"/"+fold+"/validation.txt", validation, fmt='%f')
  np.savetxt(NewDataFolder+"/"+fold+"/test.txt", test, fmt='%f')
  
def fiveFold(data):
  oneFifth = int(.2*len(data))
  tempData = {}
  for i in range(0,5):
    tempData[i] = data[i*oneFifth:(i+1)*oneFifth,:]
    
  for i in range(0,5):
    flag = False
    for j in range(0,5):
      if i==j:
        flag = True
      else:
        if j==0 or (j==1 and flag==True):
          rest = tempData[j]
        else:
          rest = np.concatenate((rest,tempData[j]), axis = 0)
         
      
    
    saveData(tempData[i], rest,str(i+1))
  
   
raw_data = open(DataFile)
dataset = np.loadtxt(raw_data, delimiter=",")
X = dataset[:,0:-1]
x_Filled = mean_X_Filler(X)
Y = dataset[:,-1:]
new_data = np.concatenate((x_Filled,Y), axis =1)
fiveFold(new_data)
