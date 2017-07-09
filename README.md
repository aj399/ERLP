Network interference through link prediction is a highly researched topic in biomedicine and computational social science and are generally quite difficult. The link prediction in itself is a quite strenuous task due to the fact there is no general accord regarding the best predictor(s) for specific data. The task is even aggravated due to frequent presence of missing values, class imbalance and noise in the data. 
One possible solution for the above stated problem is to create an ensemble of predictors that combines the predictions of multiple individual root predictors.  Ensemble predictors have produced highly improved results for many prediction tasks.  One of the key factors behind the success of ensemble predictors is its power to correct errors across many diverse root predictors and its gift to strengthen accurate predictions.  
The main handicap of ensemble predictors is to identify best possible combination of root predictors to increase the prediction accuracy. Another factor to consider is the prediction time, as excessive number of predictors may also elevate the time required for network inference. We propose an RL approach to identify the best possible predictor combination that provides optimum prediction performance with minimum time expense.  Our results show that the ensemble created by our procedure show good improvement in terms of time required for network inference compared to the complete ensemble, as well as far better prediction performance in contrast to best root predictors and even slightly superior predictive performance than the complete ensemble. 

Steps to test the generate ensemble predictor using the Reinforced Learning method( Currently testing with Gene regulatory Network present in GeneNetwork.csv  )

Step 1: Clone the repository

Step 2: Run dataPreProcessing.py program to fill in missing data and convert files into the format that it support five fold validation using the following command
python dataPreProcessing.py "FiveFoldData" "GeneNetwork.csv" 

Note: The following instructions are for testing results on one fold of data, but could be easily modified to be done for fivefold validation

Step 3: Run training.py program to create the 150 classifiers using 15 classifier algorithms
python training.py "Model" "Output/1/train.txt"

Step 4: Run vSetPreProduction.py to precalculate the predictions on test.py using the 150 predictors, it also calculates the f1score of all the classifiers
python vSetPrePrediction "Model" "Output" "1/Predictions" "FiveFoldData/1/test.txt" "fileMap.csv" "f1score.csv"

Step 5: Run QLearn.py (old RL ensemble), QLearnFA.py(OLD RL ensemble using function approximation), QLearnUpd.py(New RL), QLearnUpdGreedy.py(New RL that takes best ensemble travesrsed by the agent till now) for varying exploration rates(.01, .1, .25, .5)
python QLearn.py "FiveFoldData/1/validation.txt" "Output/1/f1score.csv" "Output/1/Predictions" "10" "Output/1/results(1)/Ensemble/QLearn/10.txt" "Output/1/results(1)/StepSize/QLearn/10.txt" "0.9" ".1" ".01" "1.0" "2000" "Output/1/results(1)/Time/QLearn/10.txt" "Output/1/results(1)/EpisodePlot/QLearn/10.png" "Output/1" 

Step 6: Run FinalEval.py
