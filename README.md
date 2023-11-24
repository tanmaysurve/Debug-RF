# FairDebugger
FairDebugger is a system which provides explanations for fairness-based biased predictions of random forest in the form of patterns or subsets of data-points in its training dataset which are contributing the most towards the bias. Fairdebugger's algorithm works on 2 pillars - Data Removal-Enabled Random Forest's (DaRE-RF - https://github.com/jjbrophy47/dare_rf) unlearning capabilities and greedy lattice tree search techniques.

# Main Algorithm
The main algorithm is written in FairDebugger.ipynb. It contains 2 classes - "Dataset" and "FairnessDebuggingUsingMachineUnlearning".
1. "Dataset" class is a template class for preprocessing dataset. Anyone wanting to use FairDebugger have to preprocess their dataset by inheriting this class and providing definitions to their template functions. Dataset needs to be preprocess to get 2 datasets - a numerical version and a purely categorical version. Numerical attributes can be converted to categorical using bucketization as per domain knowledge. Care should be taken that after both these preprocessing the number of rows in both versions of dataset is same in both training and testing. The number of columns can be different. Also indices of data intances in both versions should match.
2. "FairnessDebuggingUsingMachineUnlearning" is the main class where all the logic of FairDebugger exists. To get the explanations for fairness-based bias for you random forest model, simply call the create an instance of this call by providing it with an instance of "Dataset" class and few other arguments. The main function to call to get explanations after this is - "latticeSearchSubsets()". All the hyperparameters of the algorithm are passed as arguments to this function.

# How Good are the Explanations?
After you get the most bias-inducing subsets, you can further analyse them by looking at the positive label proportion for sensitive groups and feature importance deviations after deleting this subset from the model's training. The 2 functions which do this are - "drawInferencesFromResultSubsets()" and "getFeatureImportanceChanges()" respectively. 

# Datasets
We have provided FairDebugger's working on 3 datasets - Adult Income, German Credit and Stop Question Frisk dataset. You can check out FairDebugger's workings on these datasets by simply running the 3 files provided for their fairness debugging. Feel free to tinker around with hyperparameters and/or preprocessing to check their impact.

# Testing of FairDebugger
1. "Testing_DaRE_RF_Effect_on_Fairness.ipynb" --> Testing DaRE-RF unlearning technique's effect on fairness metric.
2. "Testing_FairDebugger_Efficiency.ipynb" --> Testing DaRE-RF's efficiency in terms of time as dataset dimensions (num_rows x num_cols) increases.
   
