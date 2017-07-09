#Example: python training.py "Model" "Output/1/train.txt"

import numpy as np
import warnings
import sys
import os
from sklearn.externals import joblib
from sklearn.utils import resample
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn import svm
from sklearn.naive_bayes import BernoulliNB
from sklearn import tree
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from imblearn.under_sampling import RandomUnderSampler
warnings.filterwarnings('ignore')

if(len(sys.argv)<2):
  print "Wrong No: of Input Parameters"
  print "Required format:"
  print "Argument 1: Path to Model Directory"
  print "Argument 2: Path to Training set file"
  sys.exit()

MDir = sys.argv[1]
TrFile = sys.argv[2]
if not os.path.exists(MDir):
  os.makedirs(MDir)
  
dataset = np.loadtxt(TrFile, dtype=float)
samples = []
rus = RandomUnderSampler()
for i in range(10):
  sample = resample(dataset,n_samples = len(dataset)/3)
  x = []
  y = []
  for elm in sample:
    x.append(elm[0:-1])
    print elm[-1]
    y.append(elm[-1])
    
  rus = RandomUnderSampler()
  X, Y = rus.fit_sample(x, y)
  AdaBoostclf = AdaBoostClassifier(n_estimators=100)
  LogRegclf = LogisticRegression(C=1e5)
  RandForclf = RandomForestClassifier(n_estimators=10)
  MLPclf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
  BernNBclf = BernoulliNB()
  DTreeclf = tree.DecisionTreeClassifier()
  GaussNBclf = GaussianNB()
  LDAclf = LinearDiscriminantAnalysis()
  LinSVCclf = LinearSVC()
  MultiNmNBclf = MultinomialNB()
  QDAclf = QuadraticDiscriminantAnalysis()
  Ridgeclf = RidgeClassifier()
  SGDclf = SGDClassifier()
  SVCclf = SVC()
  Knnclf = KNeighborsClassifier()
  AdaBoostclf.fit(X,Y)
  LogRegclf.fit(X,Y)
  RandForclf.fit(X,Y)
  MLPclf.fit(X,Y)
  BernNBclf.fit(X,Y)
  DTreeclf.fit(X,Y)
  GaussNBclf.fit(X,Y)
  LDAclf.fit(X,Y)
  LinSVCclf.fit(X,Y)
  MultiNmNBclf.fit(X,Y)
  QDAclf.fit(X,Y)
  Ridgeclf.fit(X,Y)
  SGDclf.fit(X,Y)
  SVCclf.fit(X,Y)
  Knnclf.fit(X,Y)
  joblib.dump(AdaBoostclf, MDir+'/AdaBoostclf'+str(i)+'.pkl')
  joblib.dump(LogRegclf, MDir+'/LogRegclf'+str(i)+'.pkl')
  joblib.dump(RandForclf, MDir+'/RandForclf'+str(i)+'.pkl')
  joblib.dump(MLPclf, MDir+'/MLPclf'+str(i)+'.pkl')
  joblib.dump(BernNBclf, MDir+'/BernNBclf'+str(i)+'.pkl')
  joblib.dump(DTreeclf, MDir+'/DTreeclf'+str(i)+'.pkl')
  joblib.dump(GaussNBclf, MDir+'/GaussNBclf'+str(i)+'.pkl')
  joblib.dump(LDAclf, MDir+'/LDAclf'+str(i)+'.pkl')
  joblib.dump(LinSVCclf, MDir+'/LinSVCclf'+str(i)+'.pkl')
  joblib.dump(MultiNmNBclf, MDir+'/MultiNmNBclf'+str(i)+'.pkl')
  joblib.dump(QDAclf, MDir+'/QDAclf'+str(i)+'.pkl')
  joblib.dump(Ridgeclf, MDir+'/Ridgeclf'+str(i)+'.pkl')
  joblib.dump(SGDclf, MDir+'/SGDclf'+str(i)+'.pkl')
  joblib.dump(SVCclf, MDir+'/SVCclf'+str(i)+'.pkl')
  joblib.dump(Knnclf, MDir+'/Knnclf'+str(i)+'.pkl')


