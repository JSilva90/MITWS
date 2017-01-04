from __future__ import division
#from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier

import multiprocessing as mp
import time
import pandas as pd
import numpy as np
import sys
import utils
import random


import sync_search
import async_search
import new_async
from ML_data import ML_instance


##implemented my own cross val cause on server cant use module
def getKFold(labels, folds = 5):
    indice = range(0, len(labels))
    random.seed(1234)
    random.shuffle(indice)
    train_indices = []
    test_indices = []
    for i in range(folds):
        train_indices.append([])
        test_indices.append([])
    counter = 0
    for i in indice:
        for j in range(folds):
            if counter == j:
                test_indices[j].append(i)
            else:
                train_indices[j].append(i)
        counter = (counter + 1) % folds
    return train_indices, test_indices
        
    
    
    

def prepare_data():
    data = pd.read_csv("data.csv")
    #data = pd.read_csv("new_data.csv")
    data.loc[data['Label'] == -1.0, 'Label'] = 0.0 
    labels = data["Label"].values
    data = data.drop("Label", 1)
    #data = data.drop("ID", 1)
    
    
    

    ##create indices for cross validations
    #k_folds = 10
    #skf = StratifiedKFold(n_splits = k_folds, shuffle = True)
    #train_indices = []
    #test_indices = []
        
    #for train, test in skf.split(data.values, labels):
    #    train_indices.append(train)
    #    test_indices.append(test)
    train_indices, test_indices = getKFold(labels, folds = 10)
            
    ml_instance = ML_instance(data, labels, train_indices, test_indices)
    return ml_instance

def main():
    if __name__ == '__main__':
        ml_test = prepare_data()
        features = range(len(ml_test.features)) ##we start features by using
        work = [[x] for x in features]
        n_workers = 31
        #new_async.starter(work, ml_test, n_workers)
        sync_search.master_behaviour(work, ml_test, n_workers)
        #async_search.starter(work, ml_test, n_workers)
        
        
main()



