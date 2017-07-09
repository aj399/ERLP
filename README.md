
Step 1: Convert excel file(big_net_final.xlsx) to csv file(workdata.csv)

Step 2: Run dataPreProcessing.py program to fill in missing data and convert files into the format that it support five fold validation using the following command
python dataPreProcessing.py "FiveFoldData" "workdata.csv" 

Note: The following instructions are for testing results on one fold of data, but could be easily modified to be done for fivefold validation

Step 3: Run training.py program to create the 150 classifiers using 15 classifier algorithms
python training.py "Model" "Output/1/train.txt"

Step 4: Run vSetPreProduction.py to precalculate the predictions on test.py using the 150 predictors, it also calculates the f1score of all the classifiers
python vSetPrePrediction "Model" "Output" "1/Predictions" "FiveFoldData/1/test.txt" "fileMap.csv" "f1score.csv"

Step 5: Run QLearn.py (old RL ensemble), QLearnFA.py(OLD RL ensemble using function approximation), QLearnUpd.py(New RL), QLearnUpdGreedy.py(New RL that takes best ensemble travesrsed by the agent till now) for varying exploration rates(.01, .1, .25, .5)
python QLearn.py "FiveFoldData/1/validation.txt" "Output/1/f1score.csv" "Output/1/Predictions" "10" "Output/1/results(1)/Ensemble/QLearn/10.txt" "Output/1/results(1)/StepSize/QLearn/10.txt" "0.9" ".1" ".01" "1.0" "2000" "Output/1/results(1)/Time/QLearn/10.txt" "Output/1/results(1)/EpisodePlot/QLearn/10.png" "Output/1" 

Step 6: Run FinalEval.py