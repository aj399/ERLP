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
from ensembleEnvMat import EnsembleEnv
if "../" not in sys.path:
  sys.path.append("../") 

from sklearn.kernel_approximation import RBFSampler

warnings.filterwarnings('ignore')
if(len(sys.argv)<14):
  print "Wrong No: of Input Parameters"
  print "If no of arguments less than all optional arguments would be set with their default value"
  print "Required format:"
  print "Argument 1: Path to validation set file"
  print "Argument 2: Path to predictors f1 score file"
  print "Argument 3: Path to predictors prediction file"
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
PredFile = sys.argv[3]
if not os.path.exists(opDir+"/results("+str(int(Epsilon*100))+")"):
  os.makedirs(opDir+"/results("+str(int(Epsilon*100))+")")

if not os.path.exists(opDir+"/results("+str(int(Epsilon*100))+")"+"/Ensemble/QLearnUpd"):
  os.makedirs(opDir+"/results("+str(int(Epsilon*100))+")"+"/Ensemble/QLearnUpd")
  
if not os.path.exists(opDir+"/results("+str(int(Epsilon*100))+")"+"/EpisodePlot/QLearnUpd"):
  os.makedirs(opDir+"/results("+str(int(Epsilon*100))+")"+"/EpisodePlot/QLearnUpd")
  
if not os.path.exists(opDir+"/results("+str(int(Epsilon*100))+")"+"/StepSize/QLearnUpd"):
  os.makedirs(opDir+"/results("+str(int(Epsilon*100))+")"+"/StepSize/QLearnUpd")
  
if not os.path.exists(opDir+"/results("+str(int(Epsilon*100))+")"+"/Time/QLearnUpd"):
  os.makedirs(opDir+"/results("+str(int(Epsilon*100))+")"+"/Time/QLearnUpd")
  
Q = {}
  
env = EnsembleEnv(NoPredictors, F1File, ValidFile, PredFile)
 
def predict(state, a=None):
  if not a:
    pred = {}
    sTemp = copy.copy(state)
    for i in range(NoPredictors):
      if i not in sTemp:
        sTemp.add(i)
        pred[i] = Q.get(tuple(sTemp),0.0)
        sTemp.remove(i)
      
      
    return pred
  
  else:    
    state.add(a)
    value = Q.get(tuple(sTemp),0.0)
    state.remove(a)
    return value
            
def update(s, a, y):
  s.add(a)
  Q[tuple(s)]= y

def makeEpsilonGreedyPolicy(epsilon, nStates):
  def policyFn(observation):  
    A = {}
    for i in range(nStates):
      if i not in observation:
        A[i] = epsilon
      
    keys = A.keys()
    nActions = len(A)
    for key in keys:
      A[key] /= nActions
    
    qValues = predict(observation)
    bestAction = max(qValues.iteritems(), key=operator.itemgetter(1))[0]
    A[bestAction] += (1.0 - epsilon)
    a1 = A.keys()
    b1 = A.values()
    return b1, a1
  return policyFn

def qLearning(env, discount_factor=0.9, alpha=0.1, epsilon=0.1, epsilon_decay=1.0):
  start = time.time()
  totalPredictors = env.noBasePredictors()
  counter = 0  
  result = []
  prevPolicy = []
  prevTop = 0
  prevTopState = 0
  noSteps = 0
  fp = open(stepFile,"w")
  Episodes = []
  Rewards = []
  for iEpisode in itertools.count():
    policy = makeEpsilonGreedyPolicy(epsilon * epsilon_decay**iEpisode, totalPredictors)
    state = env.start()
    highestreward = 0
    for t in itertools.count():
      noSteps = noSteps+1
      if noSteps == 100000 or noSteps == 200000 or noSteps == 300000 or noSteps == 400000 :
        fp.write(str(noSteps)+"\n")
        
      actionProbs, actions = policy(state)
      action = np.random.choice(actions, p=actionProbs)
      nextState, reward, done, _ = env.step(action)
      reward *= 1000-(len(nextState))
      if(highestreward>reward):
        reward = 0
      else:
        temp = reward
        reward = reward-highestreward
        highestreward = temp
        
      if done:
        break
             
      qValuesNext = predict(nextState)
      tdTarget = reward + discount_factor * max(qValuesNext, key=qValuesNext.get)
      update(state, action, tdTarget)   
      state = nextState
    
    Episodes.append(iEpisode)
    Rewards.append(highestreward)
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
  while len(currState)<totalPredictors:
    action = max(predict(currState).iteritems(), key=operator.itemgetter(1))[0]
    temp = max(predict(currState), key=predict(currState).get)
    currState, reward, _, _ = env.step(action)
    if temp-prevScore>top:
      top = temp-prevScore
      topState = copy.copy(currState)
      
    prevScore = temp
    step = step+1	
		
  fp.write(str(noSteps)+"\n")	
  return topState

ftime = open(timeFile,"w")
start = time.time()
result = qLearning(env, DiscountFactor, Alpha, Epsilon, EpsilonDecay )
end = time.time()
ftime.write(str(end-start))
f = open(OpFile,"w")
for i in result:
  f.write(str(i)+"\n")
  
 
f.close()
print "Result = "+str(result)
