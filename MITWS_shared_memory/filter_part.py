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

from __future__ import division
import numpy as np
import multiprocessing as mp

import utils
from myClasses import Method_settings


##based on this website http://nlp.stanford.edu/IR-book/html/htmledition/mutual-information-1.html
def calculate_mi(data_labels, bins):
    ##matrix with count of elements per label and interval of bin, columns are labels lines are classes
    quanti_matrix = create_quanti_matrix(bins, data_labels)
    total_elements = np.sum(quanti_matrix)
    
    mi = 0
    
    for i in range(0,len(quanti_matrix)): ##for every line which represents the frequency of each interval for every class
        for j in range(0, len(quanti_matrix[i])): #doesn't add any new information cause has prob = 0
            if quanti_matrix[i][j] == 0:
                continue
            aux = quanti_matrix[i][j] / total_elements
            total_line = sum(quanti_matrix[i])
            total_column = sum(quanti_matrix[:,j])
            aux_2 = quanti_matrix[i][j] * total_elements
            aux_3 = aux * (np.log2(aux_2/(total_line * total_column)))
            mi += aux_3
            
    return round(mi,8)

        
####UCLAIM Algorithm

def update_boarders(val, min_val, max_val):
    if val < min_val:
        min_val = val
    if val > max_val:
        max_val = val
    return min_val, max_val    


#receive list of data and splits it into dict separating instances by class only getting values from one feature
def divide_data_according_to_labels(data, feature):
    data_labels = {}
    min_val = 9999
    max_val = -1
    for d in data:
        val = d.values[feature]
        if d.label not in data_labels:
            data_labels[d.label] = []
        data_labels[d.label].append(val)
        min_val, max_val = update_boarders(val, min_val, max_val)
    return data_labels, min_val, max_val

##join together values from a dict on a dict, removes duplices and sorts them
def get_sorted_and_set_values(dict_vals):
    l = []
    for d in dict_vals:
        l += dict_vals[d]
    l = list(set(l))
    return l

##creates a quanti_matrix for a binarization, also returns the values separated by itnerval and label
def create_quanti_matrix(bins, data_labels):
    quanti_matrix = []
    ##create the quanti matrix, basically counts the number of examples within an interval for each class
    for d in data_labels:
        histogram, _ = np.histogram(data_labels[d], bins)
        aux = list(histogram)
        quanti_matrix.append(aux)
    ##convert into np_array
    quanti_matrix = np.array(quanti_matrix)
    return quanti_matrix

##calculates the ucaim score for a quantitization
def ucaim_score(bins, data_labels):
    quanti_matrix = create_quanti_matrix(bins, data_labels)
    #print quanti_matrix
    score = 0
    for i in range(0, len(bins)-1): ##for every interval
        interval_values = quanti_matrix[:,i]
        max_val = max(interval_values)
        if max_val == 0: ##case where interval has no points
            continue
        offset = (sum(max_val - interval_values)) / (len(data_labels) - 1)
        #offset = 1 ##use normal caim instead of UCAIM
        aux = ((max_val ** 2) * offset) / sum(interval_values)
        score += aux
    score = score / len(bins)-1
    return score
        

##calculate the CAIM quantization for a feature, receives data separated by classes
def get_ucaim_quantitization(data_labels, min_val, max_val):
    values = get_sorted_and_set_values(data_labels)
    ##get all the possible divisions
    b = []
    max_val = max_val + 0.1 ## so the max value isn't counted outside the intervals
    for i in range(0, len(values)-1):
        ##add all points that are beetween values
        new_point = values[i] + ((values[i+1] - values[i])/2)
        new_point = round(new_point,4)
        b.append(new_point)
    D = [min_val, max_val] ##quantitization list
    global_ucaim = 0.0
    k = 1
    while(True):
        ##iterate thorugh all possible division points to check which one performs better at each round
        best_score_round = 0
        best_point = 0.0
        for possible_b in b:
            if possible_b in D: ##so it doesnt test selected options
                continue
            aux_d = list(D) ##auxiliar list for testing quantization values
            aux_d.append(possible_b)
            aux_d.sort()
            aux_score = ucaim_score(aux_d, data_labels)
            if aux_score > best_score_round:
                best_score_round = aux_score
                best_point = possible_b
        if best_score_round > global_ucaim or k < len(data_labels):
            global_ucaim = best_score_round
            D.append(best_point)
            D.sort()
        else:
            break
        k += 1
        if len(D) > 3:
            print D
    return D, global_ucaim
        
def discritize_data(data, used_bins):
    ##save the discrite points for each interval
    bins_points = [] ##contem um tuplo (valores, bins) para cada feature, para discretizar os valores
    for bins in used_bins: ##for every feature
        points = []
        if bins != []:
            for i in range(len(bins)-1): #
                p = bins[i] + ((bins[i+1] - bins[i])/2)
                p = round(p, 4)
                points.append(p)
        bins_points.append((points, bins))
    ##discretize all data
    for d in data: ##for every instance
        for i in range(len(d.values)):  ##for every feature
            if bins_points[i][0] == []:
                d.values[i] = 0
                continue
            val = d.values[i] ## get the value
            aux = np.digitize([val], bins_points[i][1]) ##get the intervall in which it belongs
            if aux[0] == len(bins_points[i][1]): ##point outside upper bound                
                index = aux[0] - 2
            elif aux[0] == 0:  #3point outside lower bound
                index = 0
            else:  ##normal point
                index = aux[0]-1
            #print val, " ", aux[0], " ", bins_points[i]
            d.values[i] = bins_points[i][0][index] ##get the value to convert according to intervall
    return data
            
##parallel function to perform the filtering
def f_features(id, features, data, lock, com):
    filter_scores = []
    for ft in features: ##process every feature
        data_labels, min_val, max_val = divide_data_according_to_labels(data, ft)
        if min_val == max_val:
            filter_scores.append((ft, [], 0, 0))  ##useless feature cause it always have the same value
            continue
        
        ucaim_bins, score = get_ucaim_quantitization(data_labels, min_val, max_val)
        mi = calculate_mi(data_labels, ucaim_bins)
        filter_scores.append((ft, ucaim_bins, score, mi))
    
    ##calculated everything time to send everything to the main
    lock.acquire() ##mutual exclusion to grant coherency
    output, input = com
    if output.poll(): ##if it has a message
        msg = output.recv() ##add what already was there and append this process results
        filter_scores = msg + filter_scores
    input.send(filter_scores) ##send everything
    lock.release() ##release lock
    
    return 
        


def filter_features(data, settings):
    n_fts = len(data[0].values)
    print "Dataset has ", n_fts, " features, processing filter selection in parallel"
    #com = mp.Pipe() ##one pipe is not enough for huge datasets, must change the implementation
    
    n_proc = settings.number_proc
    pipes = []
    for n in range(n_proc):
        pipes.append(mp.Pipe())
    
    shared_lock = mp.Lock()
    
    features_division = utils.divide_features(n_proc, range(n_fts), False) ##Divide features among processes, False means to return actual number of feature instead of classes
    
    ##parallelize work
    workers = []
    for i in range(1, n_proc):
        p = mp.Process(target=f_features, args=(i, features_division[i], data, shared_lock, pipes[i]))
        workers.append(p)
        p.start()
    
    f_features(0, features_division[0], data, shared_lock, pipes[0])
    
    for w in workers: ##hopefully this waits for every process to finish...
        w.join()
    
    filter_scores = []
    for p in pipes:
        output, input = p
        aux = output.recv()
        filter_scores += aux
    
    if len(filter_scores) != n_fts: ##either we received the score of every feature or some error happened
        print "Didn't receive score for every features. ERROR"
        quit()
        
    filter_scores = sorted(filter_scores, key=lambda x:x[0])  ##order results by ft
    
    used_bins = []
    for f in filter_scores:
        used_bins.append(f[1])
    
    
    ##transform data into discritized_data, if you want to discritize all data for the wrapper part uncomment this.
    ##From my tests, it improved the wrapper ability to find a subset with high accuracy, however discretizing unseen that provided bad results.
    #data = discritize_data(data, used_bins)
    
    ##order scores by MI score descendint
    sort_by_mi = sorted(filter_scores, key=lambda x:x[3])
    sort_by_mi.reverse()
    
    #select features
    #define threshold for the mi based on the top mi score
    top_mi = sort_by_mi[0][3]
    threshold = round(top_mi - (top_mi * settings.percentage_filter), 8)
    
    fts = []
    for s in sort_by_mi:
        if s[3] >= threshold:
            fts.append(s[0])
    #for i in range(0,26):
     #   fts.append(sort_by_mi[i][0])
    return fts, used_bins, sort_by_mi
    #print sort_by_mi
    
    #print "CAIM"
    #print sort_by_ucaim
        

