#Example: python vSetPrePrediction "Model" "Output" "1/Predictions" "FiveFoldData/1/test.txt" "fileMap.csv" "f1score.csv"

from __future__ import division
import os
import sys
import numpy as np
import re
from sklearn.metrics import f1_score
from sklearn.externals import joblib

if(len(sys.argv)<6):
  print "Wrong No: of Input Parameters"
  print "Required format:"
  print "Argument 1: Model Directory"
  print "Argument 2: Output Directory"
  print "Argument 3: Path to Predicition files Directory"
  print "Argument 4: Test set file"
  print "Argument 5: Predictor to prediction list Mapper file"
  print "Argument 6: F1 score File"
  sys.exit()

MDir = sys.argv[1]
OpDir = sys.argv[2]
PredDir = sys.argv[3]
TestFile = sys.argv[4]
MapFile = sys.argv[5]
F1File = sys.argv[6]
PredSubFolder = PredDir[0] 
if not os.path.exists(OpDir):
  os.makedirs(OpDir)
  
if not os.path.exists(OpDir+"/"+PredSubFolder):
  os.makedirs(OpDir+"/"+PredSubFolder)
  
if not os.path.exists(OpDir+"/"+PredDir):
  os.makedirs(OpDir+"/"+PredDir)
  
dataset = np.loadtxt(TestFile, dtype=float)
X = dataset[:,0:-1]
y = dataset[:,-1:]
Y = []
for tempy in y:
  Y.append(tempy)

fileNum = 0
fMap = open(OpDir+"/"+PredSubFolder+"/"+MapFile, "w")
fF1score = open(OpDir+"/"+PredSubFolder+"/"+F1File, "w")
for file in os.listdir(MDir):
  if file.endswith(".pkl"):
    fileNumstr = str(fileNum)
    fMap.write(file+","+fileNumstr+"\n")
    fPred = open(OpDir+"/"+PredDir+"/"+fileNumstr+".txt", "w") 
    clf = joblib.load(MDir+"/"+file)
    predY = []
    for x in X:
      predY.append(clf.predict(x.tolist()))
    
    for y_pred in predY:
      fPred.write(str(y_pred)+"\n")
      
    fPred.close()  
    f1score = f1_score(Y, predY)#Can replace with the any accuracy measure you want to use a wieght in weighted mean
    fF1score.write(fileNumstr+","+str(f1score)+"\n")
    fileNum = fileNum+1
  
  
fMap.close()
fF1score.close()
