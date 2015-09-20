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
import os
import matplotlib
from matplotlib import pyplot as plt
#matplotlib.use('Agg')

##my modules
from myClasses import ML_set
from myClasses import data_instance
from myClasses import ML_subset

test_ids = [-2054751935, 62963719, 304102792, -1420900415, -1739028311, -1626125349]

def update_boarders(val, min, max):
    if val < min:
        min = val
    if val > max:
        max = val
    return min, max    

        
##returns two arrays with stressed and not stressed utterance values and min max of the values of some feature
def get_utterance_values_of_ith_utterance(data, index):
    feature_values_no_stress = []
    feature_values_stress = []
    min = 9999
    max = -9999
    
    for i in data:
        val = i.values[index]
        if i.stress == 1: 
            feature_values_stress.append(val)
        else:
            feature_values_no_stress.append(val)
        min, max = update_boarders(val, min, max)
    #print feature_values_no_stress, "\n\n\n", feature_values_stress
    return feature_values_no_stress, feature_values_stress, min, max    
        

##randomly divides dataset into test and train
def divide_data(data):
    train_set = []
    test_set = []
    for i in range(0,len(data)):
        r = random.randint(0,10)
        if r > 7:
            test_set.append(i)
        else:
            train_set.append(i)
    print "size trainset: ", len(train_set), " size testset: ", len(test_set)
    return train_set, test_set


def divide_data_not_random(data):
    train_set = []
    test_set = []
    for i in range(0, len(data)):
        if data[i].id in test_ids:
            test_set.append(i)
        else:
            train_set.append(i)
    print "size trainset: ", len(train_set), " size testset: ", len(test_set)
    return train_set, test_set

##given the index of train and test, this function returns the selected features from the data divided into test and train set
def cut_dataset(n_train, n_test, data, fts): ##cut the irrelevant features from the dataset
    train_list = []
    train_pred = []
    test_list = []
    test_pred = []
    ##data is not normalized here...
    
    for i in range(0, len(data)):
        aux = data[i]
        fts_vals = []
        for j in fts:
            fts_vals.append(aux.values[j])
        if i in n_test: ##check if index is on the test set
            test_list.append(fts_vals)
            test_pred.append(aux.stress)
        else:
            train_list.append(fts_vals)
            train_pred.append(aux.stress)
    test_set = ML_set(test_list, test_pred, fts)
    train_set = ML_set(train_list, train_pred, fts)
    
    return train_set, test_set




##make scatter plots, receive two dic of frequencies
def make_scatter(dic1, dic2, index):
    plt.figure(figsize=(25.0, 10.0))
    plt.hold = True
    plt.scatter(dic1.keys(), dic1.values() ,color='red', label = "stressed values", s=10)
    plt.scatter(dic2.keys(), dic2.values(), color='blue', label = "not stressed values", s=10)
    plt.xlim(-0.5 , 10.5)
    plt.legend(loc='upper right')
    #plt.yticks(values, names)
    plt.title('Feature ' + str(index + 1) + ' comparison')
    plt.ylabel('Frequency')
    plt.xlabel('Feature values')
    plt.grid()
    plt.savefig(base_dir + "plots/freq_compare_feature" + str(index + 1) + ".png")#, dpi=400)
    plt.close()
    

##make histogram to compare two distributions    
def make_hist(ns_data, s_data, index):
    plt.figure()
    plt.hist(ns_data, bins=100, histtype='stepfilled', normed=True, color='b', label='not stress')
    plt.hist(s_data, bins=100, histtype='stepfilled', normed=True, color='r', alpha=0.5, label='stress')
    plt.title("Stress / Not stress")
    plt.xlabel("Value")
    plt.ylabel("Probability")
    plt.legend()
    #plt.show()
    plt.savefig("hist_plots\plot" + str(index) + ".jpg")
    plt.close()
    #quit()
    
##make plot receies list of lists for y values and x values
def make_plot(y_values, x_values):
    plt.figure(figsize=(25.0, 10.0))
    plt.hold = True
    markers = ['sb-', 'xr-', 'sg-', 'xy-']
    labels = ["DT without test set", "DT with test set", "SVM without test set", "SVM with test set"]
    ##add data to the plot
    for i in range(0,4):
        plt.plot(x_values, x_values[i], markers[i], labels[i])
    plt.ylim(40 , 120)
    plt.legend(loc='upper right')
    plt.title('How the devision of train set and test set influences the accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Test number')
    plt.grid()
    plt.savefig("acc" + str(j) + ".png")#, dpi=400)
    plt.close()
    
##separates the files of some plots
def separate_plots(numbers):
    for root, _, filenames in os.walk(base_dir + "plots/"):
        for f in filenames:
            n = int(f.split(".")[0].split("feature")[1])
            if n in numbers:
                os.system("cp " + base_dir + "/plots/" + f + " " + base_dir + "moved_plots/")
    
    
def divide_features(n_proc, features):
    p_fts = []
    for i in range(0,n_proc):
        p_fts.append([])
    list = 0
    for i in range(0, len(features)):
        aux = ML_subset([features[i]], [])
        p_fts[list].append(aux)
        #p_fts[list].append(i)
        list += 1
        list = list % n_proc
        
    return p_fts
        