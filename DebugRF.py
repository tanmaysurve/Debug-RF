#!/usr/bin/env python
# coding: utf-8

# In[2]:


import time
import itertools as it
import dare
import copy
import math
from enum import Enum
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from collections import Counter
from sklearn.model_selection import train_test_split
import urllib.request as urllib


# In[3]:


'''Parent class for dataset preprocessing'''
class Dataset():
    def __init__(self, root = None, rootTrain = None, rootTest = None, column_names = None):
        self.root = root
        self.rootTrain = rootTrain
        self.rootTest = rootTest
        if self.root == None and (self.rootTrain == None or self.rootTest == None):
            raise ValueError("Dataset root path not provided")
        self.column_names = column_names
        self.dataset = pd.DataFrame()
        self.trainDataset = pd.DataFrame()
        self.testDataset = pd.DataFrame()
        self.__loadDataset()
        
    def __loadDataset(self):
        if not self.root == None: 
            if self.column_names == None:
                self.dataset = pd.read_csv(self.root)
            else:
                self.dataset = pd.read_csv(self.root, 
                                           names = self.column_names, 
                                           sep = ",")
        if not (self.rootTrain == None or self.rootTest == None):
            if self.column_names == None:
                self.trainDataset = pd.read_csv(self.rootTrain)
                self.testDataset = pd.read_csv(self.rootTest)
            else:
                self.trainDataset = pd.read_csv(self.rootTrain, 
                                           names = self.column_names, 
                                           sep = ",")
                self.testDataset = pd.read_csv(self.rootTest, 
                                           names = self.column_names, 
                                           sep = ",")
    
    def getDataset(self):
        pass
    
    def getDatasetWithNormalPreprocessing(self):
        pass
    
    def getDatasetWithCategorizationPreprocessing(self):
        pass


# In[4]:


'''Enum class for different fairness metric that are supported in the code'''
class FairnessMetric(Enum):
        SP = 1
        PP = 2
        EO = 3
        
'''Main class implementing FairDebugger Algorithm'''
class FairnessDebuggingUsingMachineUnlearning():
    '''
    dataloader = provided preprocessed dataset to work on (must be an instance of Dataset class)
    sensitiveAttribute = [String], [sensitive attribute name in dataset, priviledgedClassValue, protectedClassValue]
    classLabel = class label attribute name
    fairnessMetric = Fairness metric to be used for the algorithm
    '''
    def __init__(self, dataloader, sensitiveAttribute, classLabel, fairnessMetric):
        self.dataloader = dataloader
        if self.dataloader == None:
            raise ValueError("Dataset root path not provided")
        if not isinstance(self.dataloader, Dataset):
            raise ValueError("Inappropriate class for dataloader object provided")
        sensitiveAttributeDictKeys = ["Name", "Priviledged", "Protected"]
        self.sensitiveAttribute = {sensitiveAttributeDictKeys[i]: sensitiveAttribute[i] for i in range(0, len(sensitiveAttribute))}
        self.classLabel = classLabel
        if not isinstance(fairnessMetric, FairnessMetric):
            raise ValueError("Inappropriate Enum class for fairnessMetric")
        self.fairnessMetric = fairnessMetric
        self.train = pd.DataFrame()
        self.test = pd.DataFrame()
        self.trainX = pd.DataFrame()
        self.trainY = pd.DataFrame()
        self.testX = pd.DataFrame()
        self.testY = pd.DataFrame()
        self.testSensitiveAttr = pd.DataFrame()
        self.trainSensitiveAttr = pd.DataFrame()
        self.privTestIndices = None
        self.protTestIndices = None
        self.categorizedTrain = pd.DataFrame()
        self.categorizedTest = pd.DataFrame()
        self.columns = None
        self.predictions = None
        self.groundTruth = pd.DataFrame()
        self.dataStatisticalParity = None
        self.dataPredictiveParity = None
        self.equalizingOddsParity = None
        self.dataAccuracy = None
        self.attributeMap = {}
        self.literals = None
        self.min = 1
        self.max = 1000
        self.validSubsetIndexLists = []
        self.featureImportances = {}
        self.sorted_indices_for_feature_importances = None;
        self.__loadDataset()
        self.__getPredictions()
        self.__getFeatureImportances()
        self.__createAttributeMap()
        
    def getDatasetFairnessParity(self):
        if self.fairnessMetric == FairnessMetric.SP:
            return self.getDatasetStatisticalParity()
        elif self.fairnessMetric == FairnessMetric.PP:
            return self.getDatasetPredictiveParity()
        elif self.fairnessMetric == FairnessMetric.EO:
            return self.getDatasetEqualizingOddsParity()
    
    def getDatasetStatisticalParity(self):
        return self.dataStatisticalParity
    
    def getDatasetPredictiveParity(self):
        return self.dataPredictiveParity
        
    def getDatasetEqualizingOddsParity(self):
        return self.equalizingOddsParity
    
    def getAccuracy(self):
        return str(self.dataAccuracy * 100) + "%"
    
    def getLiterals(self):
        return self.literals
    
    def getAttributeMap(self):
        return self.attributeMap
        
    def __loadDataset(self):
        self.train, self.test = self.dataloader.getDatasetWithNormalPreprocessing()
        self.trainX = self.train.drop(self.classLabel, axis = 1)
        self.trainY = self.train[self.classLabel]
        self.testX = self.test.drop(self.classLabel, axis = 1)
        self.testY = self.test[self.classLabel]
        self.testSensitiveAttr = self.testX[self.sensitiveAttribute["Name"]]
        self.trainSensitiveAttr = self.trainX[self.sensitiveAttribute["Name"]]
        self.privTestIndices = np.where(self.testSensitiveAttr == self.sensitiveAttribute["Priviledged"])[0]
        self.protTestIndices = np.where(self.testSensitiveAttr == self.sensitiveAttribute["Protected"])[0]
        self.categorizedTrain, self.categorizedTest = self.dataloader.getDatasetWithCategorizationPreprocessing(decodeAttributeValues = True)
        self.columns = self.categorizedTrain.columns.values
        '''Feature scaling to standardize dataset to help model learn patterns'''
        for col in self.trainX.columns:     
            scaler = StandardScaler()     
            self.trainX[col] = scaler.fit_transform(self.trainX[col].values.reshape(-1, 1))
        for col in self.testX.columns:     
            scaler = StandardScaler()     
            self.testX[col] = scaler.fit_transform(self.testX[col].values.reshape(-1, 1)) 
    
    def __getPredictions(self):
        '''
        n_estimators = no. of trees
        max_depth = depth of trees
        k = no. thresholds to consider per attribute
        topd = no. of random node layers
        '''
        rf = dare.Forest(n_estimators = 100,
                         max_depth = 5,
                         k = 8,  
                         topd = 1,  
                         random_state = 1)
        rf.fit(self.trainX.to_numpy(), self.trainY.to_numpy())
        self.predictions = rf.predict(self.testX.to_numpy())
        self.groundTruth = self.testY
        self.dataAccuracy = accuracy_score(self.testY.to_numpy(), self.predictions)
        self.dataStatisticalParity = self.__getStatisticalParityDifference(self.privTestIndices, self.protTestIndices, self.predictions)
        self.dataPredictiveParity = self.__getPredictiveParityDifference(self.privTestIndices, self.protTestIndices, self.predictions)
        self.equalizingOddsParity = self.__getEqualizingOddsParityDifference(self.privTestIndices, self.protTestIndices, self.predictions)
    
    def __getFeatureImportances(self):
        rf = RandomForestClassifier(n_estimators = 100,
                                    max_depth = 5)
        rf.fit(self.trainX.to_numpy(), self.trainY.to_numpy())
        importances = rf.feature_importances_
        self.sorted_indices_for_feature_importances = np.argsort(importances)[::-1]
        for i, column in enumerate(self.trainX.columns):
            self.featureImportances[column] = importances[i];
        
    def __getFairnessParityDifference(self, priviledgedIndices, protectedIndices, predictions):
        if self.fairnessMetric == FairnessMetric.SP:
            return self.__getStatisticalParityDifference(priviledgedIndices, protectedIndices, predictions)
        elif self.fairnessMetric == FairnessMetric.PP:
            return self.__getPredictiveParityDifference(priviledgedIndices, protectedIndices, predictions)
        elif self.fairnessMetric == FairnessMetric.EO:
            return self.__getEqualizingOddsParityDifference(priviledgedIndices, protectedIndices, predictions) 
    
    def __getStatisticalParityDifference(self, priviledgedIndices, protectedIndices, predictions):
        y_pred_priviledged = predictions[priviledgedIndices]
        y_pred_protected = predictions[protectedIndices]
        spPri = len(np.where(y_pred_priviledged == 1)[0]) / (len(y_pred_priviledged) + 1)
        spPro = len(np.where(y_pred_protected == 1)[0]) / (len(y_pred_protected) + 1)
        return spPri - spPro
    
    def __getPredictiveParityDifference(self, priviledgedIndices, protectedIndices, predictions):
        y_pred_priviledged = predictions[priviledgedIndices]
        y_pred_protected = predictions[protectedIndices]
        ppPri = precision_score(self.groundTruth[priviledgedIndices].to_numpy(), y_pred_priviledged)
        ppPro = precision_score(self.groundTruth[protectedIndices].to_numpy(), y_pred_protected)
        return ppPri - ppPro
        
    def __getEqualizingOddsParityDifference(self, priviledgedIndices, protectedIndices, predictions):
        y_pred_priviledged = predictions[priviledgedIndices]
        y_pred_protected = predictions[protectedIndices]
        
        cnf_matrix_pri = confusion_matrix(self.groundTruth[priviledgedIndices].to_numpy(), y_pred_priviledged)
        FP_pri = (cnf_matrix_pri.sum(axis=0) - np.diag(cnf_matrix_pri)).astype(float)  
        FN_pri = (cnf_matrix_pri.sum(axis=1) - np.diag(cnf_matrix_pri)).astype(float)
        TP_pri = (np.diag(cnf_matrix_pri)).astype(float)
        TN_pri = (cnf_matrix_pri.sum() - (FP_pri + FN_pri + TP_pri)).astype(float)
        # Sensitivity, hit rate, recall, or true positive rate
        TPR_pri = TP_pri / (TP_pri + FN_pri)
        # Fall out or false positive rate
        FPR_pri = FP_pri/(FP_pri + TN_pri)
        
        cnf_matrix_pro = confusion_matrix(self.groundTruth[protectedIndices].to_numpy(), y_pred_protected)
        FP_pro = (cnf_matrix_pro.sum(axis=0) - np.diag(cnf_matrix_pro)).astype(float)  
        FN_pro = (cnf_matrix_pro.sum(axis=1) - np.diag(cnf_matrix_pro)).astype(float)
        TP_pro = (np.diag(cnf_matrix_pro)).astype(float)
        TN_pro = (cnf_matrix_pro.sum() - (FP_pro + FN_pro + TP_pro)).astype(float)
        # Sensitivity, hit rate, recall, or true positive rate
        TPR_pro = TP_pro / (TP_pro + FN_pro)
        # Fall out or false positive rate
        FPR_pro = FP_pro/(FP_pro + TN_pro)
        
        TPR_parity = TPR_pri[1] - TPR_pro[1]
        FPR_parity = FPR_pri[1] - FPR_pro[1]
        eo_parity = (TPR_parity + FPR_parity) / 2
        return eo_parity
    
    def __createAttributeMap(self):
        categorizedDf = pd.concat([self.categorizedTrain, self.categorizedTest], ignore_index=True)
        for column in categorizedDf.columns:
            if column != self.classLabel:
                for value in categorizedDf[column].unique():
                    self.attributeMap[value] = column

        self.literals = np.concatenate([categorizedDf[col].unique() for col in categorizedDf if col != self.classLabel])
        
    def __doesSubsetAlreadyExist(self, subset, existingSubsets):
        for item in existingSubsets:
            if(item.difference(subset) == set()):
                return True
        return False
    
    def __isSubsetRealistic(self, subset):
        columnDict = dict.fromkeys(self.columns, 0)
        for item in subset:
            if(columnDict[self.attributeMap[item]] > 0):
                return False
            else:
                columnDict[self.attributeMap[item]] += 1
        return True
    
    def __getSubsetInfo(self, subset, minSupport, maxSupport):
        requiredIndices = pd.Series(dtype = 'int') 
        firstItem = True
        for item in subset:
            if(firstItem == True):
                requiredIndices = (self.categorizedTrain[self.attributeMap[item]] == item)
                firstItem = False
            requiredIndices = requiredIndices & (self.categorizedTrain[self.attributeMap[item]] == item)
        subsetIndexLists = (self.categorizedTrain[requiredIndices].index.tolist())
        support =  len(subsetIndexLists) / len(self.categorizedTrain)
        isValidSubset = (support >= minSupport) and (support <= maxSupport)
        return {"isValid": isValidSubset, "indexList": subsetIndexLists, "support": support}
    
    def __evaluateSubset(self, subsetIndexList, p1Parity, p1Size, p2Parity, p2Size, compare, shouldCompareOrigParity):
        start = time.time()
        rf_temp = dare.Forest(n_estimators = 100,
                              max_depth = 5, 
                              k = 8,  
                              topd = 1,  
                              random_state = 1)
        rf_temp.fit(self.trainX.to_numpy(), self.trainY.to_numpy())
        end = time.time()
        timeElapsedToTrain = end - start
        start = time.time()
        rf_temp.delete(subsetIndexList)
        end = time.time()
        timeElapsedToDelete = end - start
        predictions = rf_temp.predict(self.testX.to_numpy())
        accuracy = accuracy_score(self.testY.to_numpy(), predictions)
        parityChildValue = self.__getFairnessParityDifference(self.privTestIndices, self.protTestIndices, predictions)
        isBetterThanParents = False
        if compare == 'normal':
            if shouldCompareOrigParity:
                isBetterThanParents = abs(parityChildValue) < abs(p1Parity) and abs(parityChildValue) < abs(p2Parity) and abs(parityChildValue) < abs(self.getDatasetFairnessParity())
            else:
                isBetterThanParents = abs(parityChildValue) < abs(p1Parity) and abs(parityChildValue) < abs(p2Parity)
        elif compare == "per_instance":
            avgChildValue = parityChildValue / len(subsetIndexList)
            avgP1Value = p1Parity / p1Size
            avgP2Value = p2Parity / p2Size
            avgDatasetParity = self.getDatasetFairnessParity() / len(self.trainX)
            if shouldCompareOrigParity:
                isBetterThanParents = abs(avgChildValue) < abs(avgP1Value) and abs(avgChildValue) < abs(avgP2Value) and abs(avgChildValue) < abs(avgDatasetParity)
            else:
                isBetterThanParents = abs(avgChildValue) < abs(avgP1Value) and abs(avgChildValue) < abs(avgP2Value)
        return {"isBetterThanParents": isBetterThanParents, "parity": parityChildValue, "accuracy": accuracy,
                "timeElapsedToTrain": timeElapsedToTrain, "timeElapsedToDelete": timeElapsedToDelete}
        
    def __expandSubsets(self, N, L):
        E = []
        if L == 0:
            for column in self.literals:
                E.append({"subset": {str(column)}, "parity": 0, "size": self.min,
                          "parent1": {"parity": self.max, "size": self.min},
                          "parent2": {"parity": self.max, "size": self.min}
                         })
            return E
        newSubsets = []
        for index1, parent1 in enumerate(N):
            for index2, parent2 in enumerate(N):
                if(index2 <= index1):
                    continue
                setIntersection = parent1["subset"].intersection(parent2["subset"])
                if(len(setIntersection) != (L - 1)):
                    continue 
                setUnion = parent1["subset"].union(parent2["subset"])
                if(self.__doesSubsetAlreadyExist(setUnion, newSubsets) == True):
                    continue
                if(self.__isSubsetRealistic(setUnion) == False):
                    continue    
                newSubsets.append(setUnion)
                E.append({"subset": setUnion, "parity": 0, "size": self.min,
                          "parent1": {"parity": parent1["parity"], "size": parent1["size"]},
                          "parent2": {"parity": parent2["parity"], "size": parent2["size"]}
                         })
        return E
    
    '''
    maxLiterals - Int, max number of literals which can be in subsets
    subsetSupportRange - Int or [Int, Int] or (Int, Int), support range in which subset should lie, [min, max],
                        If single value is provided range is [0.05, max].
    compare - {"normal", "per_instance"}
    shouldCompareOrigParity = True / False
    isPruning - True / False
    '''
    def latticeSearchSubsets(self, maxLiterals = 2, subsetSupportRange = 0.1, compare = "normal", shouldCompareOrigParity = True, isPruning = True):
        if isinstance(subsetSupportRange, (list, tuple, np.ndarray)):
            if subsetSupportRange[0] > subsetSupportRange[1]:
                raise ValueError("Min value cannot be greater than Max")
            minSupport, maxSupport = subsetSupportRange[0], subsetSupportRange[1]
        else:
            minSupport, maxSupport = 0.01, subsetSupportRange
        if maxSupport == 0.0:
            maxSupport += 0.1
        L = 0
        L_notPruning = 0
        E = self.__expandSubsets([], 0)
        N = []
        validSubsetsInfo = []
        self.validSubsetIndexLists = []
        while L <= (maxLiterals - 1):
            if isPruning:
                print("level: " + str(L))
            else:
                print("level: " + str(L_notPruning))
            for subset in E:
                subsetInfo = self.__getSubsetInfo(subset["subset"], minSupport, maxSupport)
                subset["size"] = len(subsetInfo["indexList"])
                if subsetInfo["isValid"] == False:
                    if subsetInfo["support"] > maxSupport:
                        N.append(subset)
                    else:
                        continue
                subsetEvalResult = self.__evaluateSubset(subsetInfo["indexList"],
                                                         subset["parent1"]["parity"], subset["parent1"]["size"],
                                                         subset["parent2"]["parity"], subset["parent2"]["size"],
                                                         compare, shouldCompareOrigParity)
                subset["parity"] = subsetEvalResult["parity"]
                isSubsetValid = False
                if isPruning:
                    isSubsetValid = subsetEvalResult["isBetterThanParents"] == True and subsetInfo["support"] <= maxSupport
                else:
                    isSubsetValid = subsetInfo["support"] <= maxSupport
                if isSubsetValid:
                    self.validSubsetIndexLists.append(subsetInfo["indexList"])
                    N.append(subset)
                    validSubsetsInfo.append({"subset": subset["subset"],
                                             "size": len(subsetInfo["indexList"]),
                                             "support": subsetInfo["support"],
                                             "parity": subset["parity"],
                                             "accuracy": subsetEvalResult["accuracy"],
                                             "timeElapsedToTrain": subsetEvalResult["timeElapsedToTrain"],
                                             "timeElapsedToDelete": subsetEvalResult["timeElapsedToDelete"]})
            if isPruning:
                L += 1
                E = self.__expandSubsets(N, L)
            else:
                L_notPruning += 1
                E = self.__expandSubsets(N, L_notPruning)
            if not E:
                break
        return self.__getResults(validSubsetsInfo)
        
    def __getGroundTruthValues(self):
        gt_parity = []
        gt_accuracy = []
        for indexList in self.validSubsetIndexLists:
            rf_temp = dare.Forest(n_estimators = 100,
                          max_depth = 5,
                          k = 8,  
                          topd = 1,  
                          random_state = 1)
            newX = self.trainX.drop(self.trainX.index[indexList]) 
            newY = self.trainY.drop(self.trainY.index[indexList])
            rf_temp.fit(newX.to_numpy(), newY.to_numpy())
            predictions = rf_temp.predict(self.testX.to_numpy())
            accuracy = accuracy_score(self.testY.to_numpy(), predictions)
            parity = self.__getFairnessParityDifference(self.privTestIndices, self.protTestIndices, predictions)
            gt_accuracy.append(accuracy)
            gt_parity.append(parity)
        return gt_parity, gt_accuracy
    
    def __getResults(self, validSubsetsInfo):
        gt_parity, gt_accuracy = self.__getGroundTruthValues()
        result = pd.DataFrame(columns = ["Subset", "Size", "Support", "Parity", "GT_Parity", 
                                         "Accuracy", "GT_Accuracy", "timeElapsedToTrain", 
                                         "timeElapsedToDelete", "Parity_Reduction", "Accuracy_Reduction"])
        for index, subset in enumerate(validSubsetsInfo):
            parityReduction = (abs(self.getDatasetFairnessParity()) - abs(subset["parity"])) * 100 / abs(self.getDatasetFairnessParity()) 
            accReduction = (abs(self.dataAccuracy) - abs(subset["accuracy"])) * 100 / abs(self.dataAccuracy)
            newResultEntry = [str(subset["subset"]),
                              str(subset["size"]),
                              str(subset["support"]),
                              str(subset["parity"]),
                              str(gt_parity[index]),
                              str(subset["accuracy"]),
                              str(gt_accuracy[index]),
                              str(subset["timeElapsedToTrain"]),
                              str(subset["timeElapsedToDelete"]),
                              str(parityReduction),
                              str(accReduction)]
            result.loc[len(result)] = newResultEntry
        self.getEstimatedVsGroundTruthParityGraph(result)
        self.getEstimatedVsGroundTruthAccuracyGraph(result)
        result["Parity_Reduction"] = result["Parity_Reduction"].astype(str).astype(float)
        result = result.sort_values('Parity_Reduction', ascending = False, ignore_index = True)
        return result
    
    def getEstimatedVsGroundTruthParityGraph(self, result):
        if len(result) == 0:
            return
        x = np.linspace(-10, 10, 1000)
        estParity = result['Parity'].astype(float).to_numpy()
        gtParity = result['GT_Parity'].astype(float).to_numpy()
        minEst, maxEst = estParity.min(), estParity.max()
        minGt, maxGt = gtParity.min(), gtParity.max()
        minValue = minEst if minEst < minGt else minGt
        maxValue = maxEst if maxEst > maxGt else maxGt
        plt.figure(figsize = (10,10))
        plt.title("Estimated Parity vs Ground Truth Parity")
        plt.xlabel("Estimated Parity")
        plt.ylabel("Ground Truth Parity")
        plt.xlim(minValue, maxValue)
        plt.ylim(minValue, maxValue)
        plt.plot(estParity, gtParity, 'ro')
        plt.plot(x, x + 0, '-g')
        plt.show()
        
    def getEstimatedVsGroundTruthAccuracyGraph(self, result):
        if len(result) == 0:
            return
        x = np.linspace(-10, 10, 1000)
        estAcc = result['Accuracy'].astype(float).to_numpy()
        gtAcc = result['GT_Accuracy'].astype(float).to_numpy()
        minEst, maxEst = estAcc.min(), estAcc.max()
        minGt, maxGt = gtAcc.min(), gtAcc.max()
        minValue = minEst if minEst < minGt else minGt
        maxValue = maxEst if maxEst > maxGt else maxGt
        plt.figure(figsize = (10,10))
        plt.title("Estimated Accuracy vs Ground Truth Accuracy")
        plt.xlabel("Estimated Accuracy")
        plt.ylabel("Ground Truth Accuracy")
        plt.xlim(minValue, maxValue)
        plt.ylim(minValue, maxValue)
        plt.plot(estAcc, gtAcc, 'ro')
        plt.plot(x, x + 0, '-g')
        plt.show()
        
    def drawInferencesFromResultSubsets(self, subsets, protectedName, priviledgedName):
        result = pd.DataFrame(columns = ["Subset", "Size", "Support", "SupportRange", 
                                         "Total_" + protectedName, "Total_" + priviledgedName,
                                         protectedName + "_1s", priviledgedName + "_1s", 
                                         protectedName + "_0s", priviledgedName + "_0s"])
        for i in range(0,len(subsets) + 1):
            subset = None
            subsetIndexLists = None
            supportRange = None
            if i != 0:
                subset = subsets[i-1]
                requiredIndices = pd.Series(dtype = 'int') 
                firstItem = True
                for item in subset:
                    if(firstItem == True):
                        requiredIndices = (self.categorizedTrain[self.attributeMap[item]] == item)
                        firstItem = False
                    requiredIndices = requiredIndices & (self.categorizedTrain[self.attributeMap[item]] == item)
                subsetIndexLists = (self.categorizedTrain[requiredIndices].index.tolist())
            else:
                subset = "Entire Train Dataset"
                subsetIndexLists = (self.categorizedTrain.index.tolist())
            subsetSensitiveGroupData = self.trainSensitiveAttr[subsetIndexLists]
            privIndices = np.where(subsetSensitiveGroupData == self.sensitiveAttribute["Priviledged"])[0]
            protIndices = np.where(subsetSensitiveGroupData == self.sensitiveAttribute["Protected"])[0]
            support =  len(subsetIndexLists) / len(self.categorizedTrain)
            if support >= 0 and support < 0.05: 
                supportRange = "LT5%"
            elif support >= 0.05 and support < 0.10:
                supportRange = "5-10%"
            elif support >= 0.10 and support < 0.30: 
                supportRange = "10-30%"
            elif support >= 0.30 and support < 1:
                supportRange = "GT30%"
            else: 
                supportRange = "100%"
            gt_status = self.trainY.to_numpy()
            eps = 0.00001
            protected_1s = len(np.where(gt_status[protIndices] == 1)[0])
            priviledged_1s = len(np.where(gt_status[privIndices] == 1)[0])
            protected_0s = len(np.where(gt_status[protIndices] == 0)[0])
            priviledged_0s = len(np.where(gt_status[privIndices] == 0)[0])
            newResultEntry = [str(subset),
                              str(len(subsetIndexLists)),
                              str(support),
                              str(supportRange),
                              str(len(protIndices)),
                              str(len(privIndices)),
                              str(round(protected_1s / (len(protIndices) + eps), 2)),
                              str(round(priviledged_1s / (len(privIndices) + eps), 2)),
                              str(round(protected_0s / (len(protIndices) + eps), 2)),
                              str(round(priviledged_0s / (len(privIndices) + eps), 2))]
            result.loc[len(result)] = newResultEntry
        return result
    
    def getFeatureImportanceChanges(self, subsets):
        cols = ["Subset"]
        datasetCols = self.trainX.columns
        for index in self.sorted_indices_for_feature_importances:
            cols.append(datasetCols[index])
        result = pd.DataFrame(columns = cols)
        for subset in subsets:
            requiredIndices = pd.Series(dtype = 'int') 
            firstItem = True
            for item in subset:
                if(firstItem == True):
                    requiredIndices = (self.categorizedTrain[self.attributeMap[item]] == item)
                    firstItem = False
                requiredIndices = requiredIndices & (self.categorizedTrain[self.attributeMap[item]] == item)
            subsetIndexLists = (self.categorizedTrain[requiredIndices].index.tolist())
            rf = RandomForestClassifier(n_estimators = 100,
                                        max_depth = 5)
            newX = self.trainX.drop(self.trainX.index[subsetIndexLists]) 
            newY = self.trainY.drop(self.trainY.index[subsetIndexLists])
            rf.fit(newX.to_numpy(), newY.to_numpy())
            importances = rf.feature_importances_
            newResultEntry = [str(subset)]
            for i in self.sorted_indices_for_feature_importances:
                featureImpChange = ((importances[i] - self.featureImportances[datasetCols[i]]) * 100) / self.featureImportances[datasetCols[i]]
                newResultEntry.append(str(featureImpChange))
            result.loc[len(result)] = newResultEntry
        return result


# In[ ]:




