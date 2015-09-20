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
import multiprocessing as mp
import time

##import my files
import prepare_data
import filter_part
import wrapper_part
import utils
import sys
import classification_part
from myClasses import ML_set
from myClasses import data_instance
from myClasses import Method_settings
from xml.dom import minidom


def create_smaller_dataset(data):
    ##creates a smaller dataset... warning: stupid code above but it works...
    true_vals = []
    false_vals = []
        
    for d in data:
        if len(false_vals) < 250:
            if d.label == -1.0:
                false_vals.append(d.values)
        if len(true_vals) < 250:
            if d.label == 1.0:
                true_vals.append(d.values)
        
    file_data = open("smaller_mad.data", "a")
    file_label = open("smaller_mad.labels", "a")
        
    if len(true_vals) < 250 or len(false_vals) < 250:
        print "NOT ENOUGH VALUES:", len(true_vals), len(false_vals)
        quit()
        
    for i in true_vals:
        aux = ""
        for j in i:
            aux += str(j) + " "
        aux = aux[:-1]
        aux += "\n"
        file_data.write(aux)
        file_label.write("1\n")
            
    for i in false_vals:
        aux = ""
        for j in i:
            aux += str(j) + " "
        aux = aux[:-1]
        aux += "\n"
        file_data.write(aux)
        file_label.write("-1\n")
        
    file_data.close()
    file_label.close()
    quit()

def nips_validation(data, best_subsets, mi_scores, params_norm, used_bins, cost, gamma, settings):
    dataset_name = settings.dataset_name
    ##read the 3 sets of data: train, validation and test
    if settings.dataset_type == "dense": ##dense type from NIPS
        data_train = prepare_data.import_nips_dense(settings.file_train[0], settings.file_train[1])
        data_valid = prepare_data.import_nips_dense(settings.file_valid[0], settings.file_valid[1])
        data_test = prepare_data.import_nips_dense(settings.file_test, "")
    elif settings.dataset_type == "sparse_binary": ##sparse_binary type from NIPS
        data_train = prepare_data.import_nips_sparse_binary(settings.file_train[0], settings.file_train[1], settings.number_features)
        data_valid = prepare_data.import_nips_sparse_binary(settings.file_valid[0], settings.file_valid[1], settings.number_features)
        data_test = prepare_data.import_nips_sparse_binary(settings.file_test[0], "", settings.number_features)
    elif settings.dataset_type == "sparse_integer": ##sparse_integer type from NIPS
        data_train = prepare_data.import_nips_sparse_integer(settings.file_train[0], settings.file_train[1], settings.number_features)
        data_valid = prepare_data.import_nips_sparse_integer(settings.file_valid[0], settings.file_valid[1], settings.number_features)
        data_test = prepare_data.import_nips_sparse_integer(settings.file_test[0], "", settings.number_features)
    
    ##normalize the 3 sets with the normalization parameters used during the feature selection process
    data_train = prepare_data.apply_normalization(data_train, params_norm)
    data_valid = prepare_data.apply_normalization(data_valid, params_norm)
    data_test = prepare_data.apply_normalization(data_test, params_norm)
    
    
    validation_results = {} ##save results of validation
    
    ##create variables to test and find the accuracy of the train and valid sets
    aux_data_1 = data + data_train
    folds_1 = [(range(0,len(data)), range(len(data), len(data) + len(data_train)))]
    
    aux_data_2 = data + data_valid
    folds_2 = [(range(0,len(data)), range(len(data), len(data) + len(data_valid)))]
    for i in range(0, len(best_subsets)): ##test every subset and check which generalizes best    
        acc_train = classification_part.classify(folds_1, aux_data_1, best_subsets[i][0], cost, gamma, settings.svm_kernel)
        acc_valid = classification_part.classify(folds_2, aux_data_2, best_subsets[i][0], cost, gamma, settings.svm_kernel)
        validation_results[i] = (acc_train, acc_valid)
    
    ##selection the subset that was able to obtain the best score for both sets... this could be changed
    top_score_1 = 0.0
    top_score_2 = 0.0
    top_subset = ""
    top_score = 0.0
    for i in validation_results:
        print best_subsets[i][0], validation_results[i]
        score_1 = validation_results[i][0]
        score_2 = validation_results[i][1]
        if score_1 + score_2 > top_score:
            top_score = score_1 + score_2
            top_score_1 = score_1
            top_score_2 = score_2
            top_subset = best_subsets[i][0]
        elif score_1 + score_2 == top_score: ##case where they have same percentage
            if abs(score_1 - score_2) < abs(top_score_1 - top_score_2):
                top_score = score_1 + score_2
                top_score_1 = score_1
                top_score_2 = score_2
                top_subset = best_subsets[i][0]
                
    
    print top_score_1, top_score_2 , "selected subset:", top_subset
    
    ##create the nips file for each set
    classify_data(data, top_subset, dataset_name + "_train", data_train, cost, gamma, settings.svm_kernel)
    classify_data(data, top_subset, dataset_name + "_valid", data_valid, cost, gamma, settings.svm_kernel)
    classify_data(data, top_subset, dataset_name + "_test", data_test, cost, gamma, settings.svm_kernel)
    
    ##write the selected features to the file using the MI score as sort criterion
    top_subset = order_importance_of_features(top_subset, mi_scores)
    f_fts = open("results/" + dataset_name + ".feat", "a")
    
    for ft in top_subset:
        f_fts.write(str(int(ft)+1) + "\n")
    f_fts.close()
    
def order_importance_of_features(subset, mi_scores):
    aux_list = []
    for ft in subset:
        ft = int(ft)
        for score in mi_scores:
            if score[0] == ft:
                aux_list.append((ft, score[3]))
                break
    sorted_list = sorted(aux_list, key=lambda x:x[1])
    sorted_list.reverse()
    
    ordered_ft_list = []
    for s in sorted_list:
        ordered_ft_list.append(s[0])
    return ordered_ft_list

def classify_data(data, features, base_file, data_to_classify ,cost, gamma,svm_kernel):
    
    test_classification = classification_part.classify_final_test(data, features, data_to_classify, cost, gamma, svm_kernel)
    ##finally write the results
    f_res = open("results/" + base_file + ".resu", "a")
        
    for t in test_classification:
        f_res.write(str(t) + "\n")
    f_res.close()

def read_settings():
    xml_file = "settings.xml"
    xmldoc = minidom.parse(xml_file)
    settings_list = xmldoc.getElementsByTagName('setting')
    settings = []
    for s in settings_list:
        settings.append(str(s.childNodes[0].nodeValue))
    print settings
    m_settings = Method_settings(settings)
    return m_settings

def main():
    if __name__ == '__main__':
        start_t = time.time()
        
        ##read settings from the xml file
        settings = read_settings()
        #quit()
        
        print "Using ", settings.number_proc, " processes"
        
        
        ##read data according to xml file settings
        if settings.dataset_type == "csv": ##dados separados por , sendo a ultima coluna a label
            data = prepare_data.import_data_csv(settings.file_train[0], "")
        elif settings.dataset_type == "dense": ##dense type from NIPS
            data = prepare_data.import_nips_dense(settings.file_train[0], settings.file_train[1])
        elif settings.dataset_type == "sparse_binary": ##sparse_binary type from NIPS
            data = prepare_data.import_nips_sparse_binary(settings.file_train[0], settings.file_train[1], settings.number_features)
        elif settings.dataset_type == "sparse_integer": ##sparse_integer type from NIPS
            data = prepare_data.import_nips_sparse_integer(settings.file_train[0], settings.file_train[1], settings.number_features)
        else:
            print "Not a valid option for dataset type. Current accepted values: csv, dense, sparse_binary, sparse_integer"
            quit()
        
        print "Read data with size ", len(data), " and ", len(data[0].values), " features."
        
        
        
        #create_smaller_dataset(data)
        
        ##normalize data
        params_norm = []
        data, params_norm = prepare_data.normalize_data_range(data) ##return the params used for normalization to aplly on future data
        
        ##filter the irrelevant features
        features, used_bins, mi_scores = filter_part.filter_features(data, settings) ##save the used bins to calculate future data
        
        print "selected _features:\n", features 
        
        ##call the wrapper part
        cost, gamma = wrapper_part.wrapper(data, features, settings) ##returns the used cost and gamma
        ##wrapper part is over
        print "program took: ", time.time() - start_t 
        
        ##each process saves the top 5 subsets to a file
        f_res = open("res.csv", "r")
        lines = f_res.readlines()
        f_res.close()
        
        total_nodes = 0
        removed_nodes_by_cut = 0
        wasted_time = 0.0
        send_time = 0.0
        times_request_work = 0
        results = []
        times_work_not_sent = 0
        
        for res in lines:
            res = res.split(",")
            if "PROC" in res[0]: ##ignore info lines
                total_nodes += int(res[2])
                removed_nodes_by_cut += int(res[4])
                wasted_time += float(res[5])
                send_time += float(res[6])
                times_request_work = int(res[7])
                times_work_not_sent = int(res[8])
                continue
            score = float(res[len(res)-1])
            solution = res[:len(res)-1]
            aux_solution = []
            for s in solution: ##convert to ints
                aux_solution.append(int(s))
            results.append((aux_solution, score))
            #if score > best_score:
             #   best_score = score
              #  best_set = res=[:1]
        results.sort(key=lambda tup: tup[1]) ##order by score
        results.reverse() ##Descend
        
        ##save the best subsets into a file
        outfile = open("bestsets.txt", "a")
        for res in results:
            outfile.write(str(res[0]) + "," + str(res[1]) + "\n")
        outfile.close()
        
        
        ##got results, now lets select test the validation part
        print "Tested a total of: ", total_nodes, "nodes removed by cut mec:", removed_nodes_by_cut
        print "Wasted time receiving:", wasted_time / float(settings.number_proc), " sending:", send_time/float(settings.number_proc), " requested work:", times_request_work/float(settings.number_proc), " times work not sent: ", times_work_not_sent
        print "Using c and g as parameters: ", cost, gamma
        print "best set ", results[0], " fts:", len(results[0][0])
        
        #quit()
        
        ## The validation consists in selecting the best subset that generalizes the best for unseen data. This only works in case of the nips challenge need to be adapted to different datasets
        ##mudar isto para testar todos os resultados no validation set e usar o melhr##
        nips_validation(data, results, mi_scores, params_norm, used_bins, cost, gamma, settings)
    


    
main()