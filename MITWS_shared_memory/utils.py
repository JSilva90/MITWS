'''
Copyright (C) <2015>  <Jorge Silva> <up201007483@alunos.dcc.fc.up.pt>

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>

This work was partially supported by national funds through project VOCE 
(PTDC/EEA-ELC/121018/2010), and in the scope of R&D Unit UID/EEA/50008/2013, 
funded by FCT/MEC through national funds and when applicable cofunded
by FEDER/PT2020 partnership agreement.
'''

import random

##my modules
from myClasses import ML_set
from myClasses import data_instance
from myClasses import ML_subset

test_ids = [-2054751935, 62963719, 304102792, -1420900415, -1739028311, -1626125349]




##division for voce dataset
def divide_data_not_random(data):
    test_ids = [-2054751935, 62963719, 304102792, -1420900415, -1739028311, -1626125349]
    train_set = []
    test_set = []
    for i in range(0, len(data)):
        if data[i].id in test_ids:
            test_set.append(i)
        else:
            train_set.append(i)
    #print "size trainset: ", len(train_set), " size testset: ", len(test_set)
    return train_set, test_set

##given the index of train and test, this function returns the selected features from the data divided into test and train set
def cut_dataset(n_train, n_test, data, fts): ##cut the irrelevant features from the dataset
    train_list = []
    train_pred = []
    test_list = []
    test_pred = []
 
    
    for i in range(0, len(data)):
        aux = data[i]
        fts_vals = []
        for j in fts:
            fts_vals.append(aux.values[j])
        if i in n_test: ##check if index is on the test set
            test_list.append(fts_vals)
            test_pred.append(aux.label)
        else:
            train_list.append(fts_vals)
            train_pred.append(aux.label)
    test_set = ML_set(test_list, test_pred, fts)
    train_set = ML_set(train_list, train_pred, fts)
    
    return train_set, test_set


##separate features for every processor
def divide_features(n_proc, features, return_class):
    p_fts = []
    for i in range(0,n_proc):
        p_fts.append([])
    list = 0
    for i in range(0, len(features)):
        if return_class:
            aux = ML_subset([features[i]], [])
        else:
            aux = i
        p_fts[list].append(aux)
        #p_fts[list].append(i)
        list += 1
        list = list % n_proc
    return p_fts

##divides a list into n_proc parts, division is as balanced as possible
def divide_list(n_proc, l):
    division = []
    for i in range(0, n_proc):
        division.append([])
    aux = 0
    for i in range(0, len(l)):
        division[aux].append(l[i])
        aux += 1
        aux = aux % n_proc
    return division
        
def generate_random_sets(features, n_tests, min_size, max_size):
    test_sets = []
    while len(test_sets) < n_tests:
        sample_size = random.randint(min_size, max_size) ## generate a random size for the set
        
        #solution by ninjagecko @ http://stackoverflow.com/questions/6482889/get-random-sample-from-list-while-maintaining-ordering-of-items
        indices = random.sample(range(len(features)), sample_size)
        aux = [features[i] for i in sorted(indices)]
        if aux not in test_sets:
            test_sets.append(aux)
    return test_sets


