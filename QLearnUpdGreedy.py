import itertools
import matplotlib
import numpy as np
import sys
import copy
import operator
import os
import time
import matplotlib.pyplot as plt
from ensembleEnvMat import EnsembleEnv
if "../" not in sys.path:
  sys.path.append("../") 

from sklearn.kernel_approximation import RBFSampler

if(len(sys.argv)<14):
  print "Wrong No: of Input Parameters"
  print "If no of arguments less than all optional arguments would be set with their default value"
  print "Required format:"
  print "Argument 1: Path to validation set file"
  print "Argument 2: Path to predictors f1 score file"
  print "Argument 3: Path to predictors prediction file"
  print "Argument 4: No: of predictors"
  print "Argument 5: Path to output file"
  print "Argument 6: Path to step file"
  print "Argument 7: Discount factor"
  print "Argument 8: Alpha"
  print "Argument 9: Epsilon"
  print "Argument 10: Epsilon Decay"
  print "Argument 11: Total Steps"
  print "Argument 12: Time File"
  print "Argument 13: Plot File"
  print "Argument 14: output"
  sys.exit()

ValidFile = sys.argv[1]
F1File = sys.argv[2]
PredFile = sys.argv[3]
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
 
def predict(state):
  pred = {}
  sTemp = copy.copy(state)
  for i in range(NoPredictors):
    if i not in sTemp:
      sTemp.add(i)
      pred[i] = Q.get(tuple(sTemp),0.0)
      sTemp.remove(i)
    
    
  return pred  
            
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
  noSteps = 0
  fp = open(stepFile,"w")
  Episodes = []
  Rewards = []
  highestReward = 0
  highestState = {}
  for iEpisode in itertools.count():
    policy = makeEpsilonGreedyPolicy(epsilon * epsilon_decay**iEpisode, totalPredictors)
    state = env.start()
    highestreward = 0
    for t in itertools.count():
      noSteps = noSteps+1
      fp.write(str(noSteps)+"\n")
        
      actionProbs, actions = policy(state)
      action = np.random.choice(actions, p=actionProbs)
      nextState, reward, done, _ = env.step(action)
      reward = reward*1000-(len(nextState))
      if(reward> highestReward):
        highestReward = reward
        highestState = copy.copy(nextState)
        
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
    if noSteps > TotalSteps:
      break
      
  plt.plot(Episodes, Rewards)
  plt.xlabel('Episodes')
  plt.ylabel('Episode Reward')
  plt.title('Episode Reward over time')
  plt.savefig(plotFile)
  fp.write(str(noSteps)+"\n")	
  return highestState

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
