import itertools
import matplotlib
import numpy as np
import sys
import copy
import sklearn.pipeline
import sklearn.preprocessing
import random
import operator
import warnings
import os
import time
import matplotlib.pyplot as plt
from ensembleEnv import EnsembleEnv
if "../" not in sys.path:
  sys.path.append("../") 

from sklearn.linear_model import SGDRegressor
from sklearn.kernel_approximation import RBFSampler

warnings.filterwarnings('ignore')
if(len(sys.argv)<14):
  print "Wrong No: of Input Parameters"
  print "If no of arguments less than all optional arguments would be set with their default value"
  print "Required format:"
  print "Argument 1: Path to validation set file"
  print "Argument 2: Path to predictors f1 score file"
  print "Argument 3: Path to predictors prediction directory"
  print "Argument 4: No: of predictors(optional = 10)"
  print "Argument 5: Path to output file(optional = QLearn.txt)"
  print "Argument 6: Path to step file(optional = stepQL.txt)"
  print "Argument 7: Discount factor(optional = 0.9)"
  print "Argument 8: Alpha(optional = .1)"
  print "Argument 9: Epsilon(optional = .01)"
  print "Argument 10: Epsilon Decay(optional = 1.0)"
  print "Argument 11: Steps(optional = 500)"
  print "Argument 12: Time File(optional = time.txt)"
  print "Argument 13: Plot File(optional = plot.txt)"
  print "Argument 14: output(optional = plot.txt)"
  sys.exit()
else:
  NoPredictors = int(sys.argv[4])
  OpFile = sys.argv[5]
  stepFile = sys.argv[6]
  DiscountFactor = float(sys.argv[7])
  Alpha = float(sys.argv[8])
  Epsilon = float(sys.argv[9])
  EpsilonDecay = float(sys.argv[10])
  TotalSteps = int(sys.argv[11])
  timeFile = sys.argv[12]
  plotFile = sys.argv[13]
  opDir = sys.argv[14]

ValidFile = sys.argv[1]
F1File = sys.argv[2]
PredDir = sys.argv[3]
if not os.path.exists(opDir+"/results("+str(int(Epsilon*100))+")"):
  os.makedirs(opDir+"/results("+str(int(Epsilon*100))+")")

if not os.path.exists(opDir+"/results("+str(int(Epsilon*100))+")"+"/Ensemble/QLearnFA"):
  os.makedirs(opDir+"/results("+str(int(Epsilon*100))+")"+"/Ensemble/QLearnFA")
  
if not os.path.exists(opDir+"/results("+str(int(Epsilon*100))+")"+"/EpisodePlot/QLearnFA"):
  os.makedirs(opDir+"/results("+str(int(Epsilon*100))+")"+"/EpisodePlot/QLearnFA")
  
if not os.path.exists(opDir+"/results("+str(int(Epsilon*100))+")"+"/StepSize/QLearnFA"):
  os.makedirs(opDir+"/results("+str(int(Epsilon*100))+")"+"/StepSize/QLearnFA")
  
if not os.path.exists(opDir+"/results("+str(int(Epsilon*100))+")"+"/Time/QLearnFA"):
  os.makedirs(opDir+"/results("+str(int(Epsilon*100))+")"+"/Time/QLearnFA")
InitialState ={}
for i in range(NoPredictors):
  InitialState[i] = False
  
env = EnsembleEnv(NoPredictors, F1File, ValidFile, PredDir)

class Estimator():
  def __init__(self):
    self.model = SGDRegressor(learning_rate="constant") 
    self.model.partial_fit((env.start()).values(),[0])
    
  def predict(self, state, a=None):
    if not a:
      pred = {}
      sTemp = copy.copy(state)
      for i in range(len(sTemp)):
        if sTemp[i] == False:
          sTemp[i] = True
          pred[i] = self.model.predict(sTemp.values())
          sTemp[i] = False
        
      
      return pred
    
    else:    
      state[a] = True
      return self.model.predict(s.values())
            
  def update(self, s, a, y):
    s[a] = True
    self.model.partial_fit(s.values(), [y])

def finish(state, nPredictors):
  for i in range(nPredictors):
    if state[i] == False:
      return True
    
  
  return False

def makeEpsilonGreedyPolicy(estimator, epsilon, nStates):
  def policyFn(observation):  
    A = {}
    for i in range(nStates):
      if observation[i] == False:
        A[i] = epsilon
      
    keys = A.keys()
    nActions = len(A)
    for key in keys:
      A[key] /= nActions
    
    qValues = estimator.predict(observation)
    bestAction = max(qValues.iteritems(), key=operator.itemgetter(1))[0]
    A[bestAction] += (1.0 - epsilon)
    a1 = A.keys()
    b1 = A.values()
    return b1, a1
  return policyFn

def qLearning(env, estimator, discount_factor=0.9, alpha=0.1, epsilon=0.1, epsilon_decay=1.0):
  start = time.time()
  totalPredictors = env.noBasePredictors()
  counter = 0  
  result = []
  prevPolicy = []
  prevTop = 0
  prevTopState = 0
  noSteps = 0
  Episodes = []
  Rewards = []
  fp = open(stepFile,"w")
  for iEpisode in itertools.count():
    policy = makeEpsilonGreedyPolicy(estimator, epsilon * epsilon_decay**iEpisode, totalPredictors)
    state = env.start()
    tReward = 0
    for t in itertools.count():
      noSteps = noSteps+1
      if noSteps == 100000 or noSteps == 200000 or noSteps == 300000 or noSteps == 400000 :
        fp.write(str(noSteps)+"\n")
        
      actionProbs, actions = policy(state)
      action = np.random.choice(actions, p=actionProbs)
      
      nextState, reward, done, _ = env.step(action)
      tReward += reward
      if done:
        break
             
      qValuesNext = estimator.predict(nextState)
      tdTarget = reward + discount_factor * max(qValuesNext, key=qValuesNext.get)
      estimator.update(state, action, tdTarget)   
      state = nextState
      print noSteps
    
    tReward /= totalPredictors
    Episodes.append(iEpisode)
    Rewards.append(tReward)  
    if noSteps > TotalSteps or (time.time()-start) > 561600:
      break
      
  plt.plot(Episodes, Rewards)
  plt.xlabel('Episodes')
  plt.ylabel('Episode Reward')
  plt.title('Episode Reward over time')
  plt.savefig(plotFile)  
  top = 0
  topState = []
  currState = env.start()
  step = 0
  prevReward = 0
  prevState = {}
  prevScore = 0
  while finish(currState, totalPredictors):
    action = max(estimator.predict(currState).iteritems(), key=operator.itemgetter(1))[0]
    temp = max(estimator.predict(currState), key=estimator.predict(currState).get)
    currState, reward, _, _ = env.step(action)
    if temp-prevScore>top:
      top = temp-prevScore
      topState = copy.copy(currState)
      
    prevScore = temp
    step = step+1	
		
  fp.write(str(noSteps)+"\n")	
  return topState


ftime = open(timeFile,"w")
estimator = Estimator()
start = time.time()
result = qLearning(env, estimator, DiscountFactor, Alpha, Epsilon, EpsilonDecay )
end = time.time()
ftime.write(str(end-start))
f = open(OpFile,"w")
for i in range(NoPredictors):
  if result[i] == True:
    f.write(str(i)+"\n")
  
 
f.close()
print "Result = "+str(result)
