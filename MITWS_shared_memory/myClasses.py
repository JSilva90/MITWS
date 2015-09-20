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

class data_instance:
    def __init__(self, i, vals, l):
        self.info = i
        self.values = vals
        self.label = l
        
        
class ML_set:
    def __init__(self, f_vals, f_preds, f_numbers):
        self.fts_values = f_vals
        self.fts_pred = f_preds
        self.fts_numbers = f_numbers
        
        
class ML_subset:
    def __init__(self, fts, scores):
        self.features = fts
        self.parents_scores = scores
        
        
class Method_settings:
    def __init__(self, list):
        self.number_proc = int(list[0])
        self.dataset_name = list[1]
        self.file_train = list[2].split(",")
        self.file_valid = list[3].split(",")
        self.file_test = list[4]
        self.dataset_type = list[5]
        self.grid_tests = int(list[6])
        self.prob_tests = int(list[7])
        self.number_features = int(list[8])
        self.percentage_filter = float(list[9])
        self.change_search_size = int(list[10])
        self.search_threshold = float(list[11])
        self.cross_validation = list[12]
        self.search_cutting_time = int(list[13])
        self.svm_kernel = list[14]
        if list[15] == "none":
            self.svm_parameters = []
        else: 
            self.svm_parameters = list[15].split(",")
        if str(list[16]) == "no":
            self.estimate_probs = False
        else:
            self.estimate_probs = True