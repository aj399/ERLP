import numpy as np
import sys
import re
import copy
import random
from sets import Set
import warnings
from sklearn.metrics import roc_auc_score

warnings.filterwarnings('ignore')

class EnsembleEnv():

  def start(self):
    self.s.clear() 
    return copy.copy(self.s)
    
  def step(self, action):
    transitions  = self.calculateTransitionProb(self.s,action)
    flag, state, reward, isdone = transitions
    self.s = state
    return [state, reward, isdone, flag]

  def fMax(self, predictors):
    predictedOutput = dict.fromkeys(range(self.totalPredictions),0)
    for predictor in predictors:
      for i,prediction in enumerate(self.predictions[predictor]):
        if prediction == 0:
          prediction = -1
				
        predictedOutput[i] = predictedOutput[i]+(float(prediction)*self.fscores[predictor])
			
		
    predOp = [] 
    for i in xrange(len(predictedOutput)):
      if(predictedOutput[i]<=0):
        predOp.append(0)
      else:
        predOp.append(1)
			 	
    reward = roc_auc_score(self.Output,predOp)
    return reward

  def calculateTransitionProb(self, current, predictor):
    current.add(predictor)
    new_state = current
    reward = self.fMax(new_state)
    isdone = False
    if(len(new_state) == self.BPredictors):
      isdone = True 
 
    return [1.0, new_state, reward, isdone]
    

  def __init__(self, NumPredictors, F1File, ValidData, PredDir):
    self.BPredictors = NumPredictors
    self.f1File = F1File
    self.predDir = PredDir
    self.s = Set()
    fFscore = open(self.f1File)
    self.fscores = dict.fromkeys(range(self.BPredictors),0)
    for i,line in enumerate(fFscore):
      line = line.rstrip('\n')
      items = re.split(r',',line)
      self.fscores[i] = float(items[1])

      
    fFscore.close()
    self.predictions = dict.fromkeys(range(self.BPredictors),0)
    self.totalPredictions = 0
    for predictor in range(self.BPredictors):
      self.totalPredictions = 0
      fPred = open(self.predDir+"/"+str(predictor)+".txt")
      Pred = []
      for prediction in fPred:
        self.totalPredictions = self.totalPredictions+1
        Pred.append(int(prediction[2]))

      fPred.close()
      self.predictions[predictor] = Pred

    raw_data = open(ValidData)
    dataset = np.loadtxt(raw_data, dtype=float)
    y = dataset[:,-1]
    self.Output = []
    for tempy in y:
      self.Output.append(int(tempy))
      
    
			 
			
		