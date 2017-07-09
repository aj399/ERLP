from __future__ import division
import os
import sys
import numpy as np
import warnings
import time
import re
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from sklearn.externals import joblib
warnings.filterwarnings('ignore')


def predList(predFile):
  fPred = open(predFile)
  Pred = []
  for line in fPred:
    line = line.rstrip('\n')
    Pred.append(int(line))
  
  return Pred
  
def prediction(Pred, clfs, fscores, X, Y):
  start = time.time()
  predY = []
  npredY = []
  for x in X:
    prediction = 0
    for pred in Pred:
      predValue = 1
      if clfs[pred].predict(x.tolist()) == 0:
        predValue = -1
      
      prediction += predValue*fscores[pred]
    
    npredY.append(prediction)
    if prediction >0:
      prediction = 1
    else:
      prediction = 0
      
    predY.append(prediction)  
  
  end = time.time()
  return {'auc':roc_auc_score(Y, predY), 'time':end-start}

raw_data = open("FiveFoldData\1\test.txt")
dataset = np.loadtxt(raw_data, dtype=float)
X = dataset[:,0:-1]
y = dataset[:,-1]
Y = []
for tempy in y:
  Y.append(int(tempy))
  
fileNum = 0
clfs = {}
for file in os.listdir("Model/1/"):
  if file.endswith(".pkl"):
    fileNumstr = str(fileNum)
    clf = joblib.load("Model/1/"+file)
    clfs[fileNum] = clf
    fileNum += 1

BPPredval = 0.0
BPPred = []
fFscore = open("Output/1/f1score.csv")
fscores = dict.fromkeys(range(150),0)
for i,line in enumerate(fFscore):
  line = line.rstrip('\n')
  items = re.split(r',',line)
  fscores[i] = float(items[1])
  if fscores[i] > BPPredval:
    BPPredval = fscores[i]
    BPPred = [i]

ensembles = ('RL_New', 'RL_NewGd', 'RL_Old', 'RL_OldFA', 'FE', 'BP')    
if os.path.exists("Output/1/results(1)"):
  aucScores = []
  predTimes = []
  RL_New = "Output/1/results(1)/Ensemble/QLearnUpd/150.txt"
  RL_Old = "Output/1/results(1)/Ensemble/QLearn/150.txt"
  RL_NewGd = "Output/1/results(1)/Ensemble/QLearnUpdGreedy/150.txt"
  RL_OldFA = "Output/1/results(1)/Ensemble/QLearnFA/150.txt"
  RL_NewPred = predList(RL_New)
  RL_NewPrediction = prediction(RL_NewPred, clfs, fscores, X, Y)
  aucScores.append(RL_NewPrediction['auc'])
  predTimes.append(RL_NewPrediction['time'])
  RL_NewPredGd = predList(RL_NewGd)
  RL_NewGdPrediction = prediction(RL_NewPredGd, clfs, fscores, X, Y)
  aucScores.append(RL_NewGdPrediction['auc'])
  predTimes.append(RL_NewGdPrediction['time'])
  RL_OldPred = predList(RL_Old)
  RL_OldPrediction = prediction(RL_OldPred, clfs, fscores, X, Y)
  aucScores.append(RL_OldPrediction['auc'])
  predTimes.append(RL_OldPrediction['time'])
  RL_OldPredFA = predList(RL_OldFA)
  RL_OldFAPrediction = prediction(RL_OldPredFA, clfs, fscores, X, Y)
  aucScores.append(RL_OldFAPrediction['auc'])
  predTimes.append(RL_OldFAPrediction['time'])
  FEPred = []
  for i in range(0,150):
    FEPred.append(i)

  FEPrediction = prediction(FEPred, clfs, fscores, X, Y)
  aucScores.append(FEPrediction['auc'])
  predTimes.append(FEPrediction['time'])
  BPPrediction = prediction(BPPred, clfs, fscores, X, Y)

  if not os.path.exists("Output/1/results(1)/Result"):
    os.makedirs("Output/1/results(1)/Result")  

  aucScores.append(BPPrediction['auc'])
  predTimes.append(BPPrediction['time'])
  yPos = np.arange(len(ensembles))
  plt.bar(yPos, aucScores, align='center', alpha=0.5)
  plt.xticks(yPos, ensembles)
  plt.ylabel('auESC')
  plt.title('ROC Performance')  
  plt.savefig("Output/1/results(1)/Result/FIG1.png")
  plt.close()

  plt.bar(yPos, predTimes, align='center', alpha=0.5)
  plt.xticks(yPos, ensembles)
  plt.ylabel('Time')
  plt.title('Time Performance')  
  plt.savefig("Output/1/results(1)/Result/FIG2.png")
  plt.close()
  
if os.path.exists("Output/1/results(10)"):
  aucScores = []
  predTimes = []
  RL_New = "Output/1/results(10)/Ensemble/QLearnUpd/150.txt"
  RL_Old = "Output/1/results(10)/Ensemble/QLearn/150.txt"
  RL_NewGd = "Output/1/results(10)/Ensemble/QLearnUpdGreedy/150.txt"
  RL_OldFA = "Output/1/results(10)/Ensemble/QLearnFA/150.txt"
  RL_NewPred = predList(RL_New)
  RL_NewPrediction = prediction(RL_NewPred, clfs, fscores, X, Y)
  aucScores.append(RL_NewPrediction['auc'])
  predTimes.append(RL_NewPrediction['time'])
  RL_NewPredGd = predList(RL_NewGd)
  RL_NewGdPrediction = prediction(RL_NewPredGd, clfs, fscores, X, Y)
  aucScores.append(RL_NewGdPrediction['auc'])
  predTimes.append(RL_NewGdPrediction['time'])
  RL_OldPred = predList(RL_Old)
  RL_OldPrediction = prediction(RL_OldPred, clfs, fscores, X, Y)
  aucScores.append(RL_OldPrediction['auc'])
  predTimes.append(RL_OldPrediction['time'])
  RL_OldPredFA = predList(RL_OldFA)
  RL_OldFAPrediction = prediction(RL_OldPredFA, clfs, fscores, X, Y)
  aucScores.append(RL_OldFAPrediction['auc'])
  predTimes.append(RL_OldFAPrediction['time'])
  FEPred = []
  for i in range(0,150):
    FEPred.append(i)

  FEPrediction = prediction(FEPred, clfs, fscores, X, Y)
  aucScores.append(FEPrediction['auc'])
  predTimes.append(FEPrediction['time'])
  BPPrediction = prediction(BPPred, clfs, fscores, X, Y)

  if not os.path.exists("Output/1/results(10)/Result"):
    os.makedirs("Output/1/results(10)/Result")  

  aucScores.append(BPPrediction['auc'])
  predTimes.append(BPPrediction['time'])
  yPos = np.arange(len(ensembles))
  plt.bar(yPos, aucScores, align='center', alpha=0.5)
  plt.xticks(yPos, ensembles)
  plt.ylabel('auESC')
  plt.title('ROC Performance')  
  plt.savefig("Output/1/results(10)/Result/FIG1.png")
  plt.close()

  plt.bar(yPos, predTimes, align='center', alpha=0.5)
  plt.xticks(yPos, ensembles)
  plt.ylabel('Time')
  plt.title('Time Performance')  
  plt.savefig("Output/1/results(10)/Result/FIG2.png")
  plt.close()

if os.path.exists("Output/1/results(25)"):  
  aucScores = []
  predTimes = []
  RL_New = "Output/1/results(25)/Ensemble/QLearnUpd/150.txt"
  RL_Old = "Output/1/results(25)/Ensemble/QLearn/150.txt"
  RL_NewGd = "Output/1/results(25)/Ensemble/QLearnUpdGreedy/150.txt"
  RL_OldFA = "Output/1/results(25)/Ensemble/QLearnFA/150.txt"
  RL_NewPred = predList(RL_New)
  RL_NewPrediction = prediction(RL_NewPred, clfs, fscores, X, Y)
  aucScores.append(RL_NewPrediction['auc'])
  predTimes.append(RL_NewPrediction['time'])
  RL_NewPredGd = predList(RL_NewGd)
  RL_NewGdPrediction = prediction(RL_NewPredGd, clfs, fscores, X, Y)
  aucScores.append(RL_NewGdPrediction['auc'])
  predTimes.append(RL_NewGdPrediction['time'])
  RL_OldPred = predList(RL_Old)
  RL_OldPrediction = prediction(RL_OldPred, clfs, fscores, X, Y)
  aucScores.append(RL_OldPrediction['auc'])
  predTimes.append(RL_OldPrediction['time'])
  RL_OldPredFA = predList(RL_OldFA)
  RL_OldFAPrediction = prediction(RL_OldPredFA, clfs, fscores, X, Y)
  aucScores.append(RL_OldFAPrediction['auc'])
  predTimes.append(RL_OldFAPrediction['time'])
  aucScores.append(FEPrediction['auc'])
  predTimes.append(FEPrediction['time'])

  if not os.path.exists("Output/1/results(25)/Result"):
    os.makedirs("Output/1/results(25)/Result")  

  aucScores.append(BPPrediction['auc'])
  predTimes.append(BPPrediction['time'])
  plt.bar(yPos, aucScores, align='center', alpha=0.5)
  plt.xticks(yPos, ensembles)
  plt.ylabel('auESC')
  plt.title('ROC Performance')  
  plt.savefig("Output/1/results(25)/Result/FIG1.png")
  plt.close()

  plt.bar(yPos, predTimes, align='center', alpha=0.5)
  plt.xticks(yPos, ensembles)
  plt.ylabel('Time')
  plt.title('Time Performance')  
  plt.savefig("Output/1/results(25)/Result/FIG2.png")
  plt.close()
  
if os.path.exists("Output/1/results(50)"):
  aucScores = []
  predTimes = []
  RL_New = "Output/1/results(50)/Ensemble/QLearnUpd/150.txt"
  RL_Old = "Output/1/results(50)/Ensemble/QLearn/150.txt"
  RL_NewGd = "Output/1/results(50)/Ensemble/QLearnUpdGreedy/150.txt"
  RL_OldFA = "Output/1/results(50)/Ensemble/QLearnFA/150.txt"
  RL_NewPred = predList(RL_New)
  RL_NewPrediction = prediction(RL_NewPred, clfs, fscores, X, Y)
  aucScores.append(RL_NewPrediction['auc'])
  predTimes.append(RL_NewPrediction['time'])
  RL_NewPredGd = predList(RL_NewGd)
  RL_NewGdPrediction = prediction(RL_NewPredGd, clfs, fscores, X, Y)
  aucScores.append(RL_NewGdPrediction['auc'])
  predTimes.append(RL_NewGdPrediction['time'])
  RL_OldPred = predList(RL_Old)
  RL_OldPrediction = prediction(RL_OldPred, clfs, fscores, X, Y)
  aucScores.append(RL_OldPrediction['auc'])
  predTimes.append(RL_OldPrediction['time'])
  RL_OldPredFA = predList(RL_OldFA)
  RL_OldFAPrediction = prediction(RL_OldPredFA, clfs, fscores, X, Y)
  aucScores.append(RL_OldFAPrediction['auc'])
  predTimes.append(RL_OldFAPrediction['time'])
  FEPred = []
  for i in range(0,150):
    FEPred.append(i)

  FEPrediction = prediction(FEPred, clfs, fscores, X, Y)
  aucScores.append(FEPrediction['auc'])
  predTimes.append(FEPrediction['time'])
  BPPrediction = prediction(BPPred, clfs, fscores, X, Y)

  if not os.path.exists("Output/1/results(50)/Result"):
    os.makedirs("Output/1/results(50)/Result")  

  aucScores.append(BPPrediction['auc'])
  predTimes.append(BPPrediction['time'])
  yPos = np.arange(len(ensembles))
  plt.bar(yPos, aucScores, align='center', alpha=0.5)
  plt.xticks(yPos, ensembles)
  plt.ylabel('auESC')
  plt.title('ROC Performance')  
  plt.savefig("Output/1/results(50)/Result/FIG1.png")
  plt.close()

  plt.bar(yPos, predTimes, align='center', alpha=0.5)
  plt.xticks(yPos, ensembles)
  plt.ylabel('Time')
  plt.title('Time Performance')  
  plt.savefig("Output/1/results(50)/Result/FIG2.png")
  plt.close()
