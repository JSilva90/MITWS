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
import time
import multiprocessing as mp
import random

from collections import Counter
##my modules
from myClasses import ML_set
from myClasses import data_instance
from myClasses import ML_subset
from myClasses import Method_settings
import utils
import classification_part
import options_wrapper


#use_probabilities = True
#leave_one_out = False ## if this is true then n_folds = len(instancances)
#grid_tests = 10
#prob_tests = 10
#cross_val = True
#n_folds = 100
#subset_size_to_switch_search = 8  ##size of subsets in which we start searching in depth and change the threshold parameters
#threshold_1 = 0.5 ##threshold for the first_part
#threshold_2 = 0.0 ##threshold for second part, expanding nodes while score increses or decreases less than threshold_2, if 0 is defined then expand only if it improves accuracy
#cutting_time = 900 ## number of seconds till each process start cuts datasets

##generate folds for classificaton
def generate_folds(data, cross_val):
    if cross_val == "leave_one_out":
        n_folds = len(data)
    else:
        n_folds = int(cross_val)
    folds = classification_part.get_folds(len(data), n_folds)
    print "Using cross_validation with ", n_folds, " folds"
    return folds



def wrapper(data, features, settings):
    print "Starting Wrapper Part With", len(features), " features"
    
    n_proc = settings.number_proc
    ##divide features among processors
    p_work_list = utils.divide_features(n_proc, features, True)
    
    ##get the division of data into train and test set, cross_validation techniques are used
    folds = generate_folds(data, settings.cross_validation)
    
    ##define if we need to estimate second parameter depending on the used kernel
    if settings.svm_kernel == "linear": ##if we are using a
        estimate_g = False
    else:
        estimate_g = True
        
    ##parameter estimation in case paramenters were not provided
    if settings.svm_parameters == []:
        print "Starting grid search with ", settings.grid_tests
        cost, gamma = classification_part.parallel_grid_search(settings, folds, data, features, estimate_g)
    else:
        cost = float(settings.svm_parameters[0])
        if estimate_g:
            gamma = float(settings.svm_parameters[1])
        else:
            gamma = 0.0
        
    print "Using ", cost, gamma, " as parameters for SVM"
    
    
    probs = options_wrapper.generate_probabilities(settings, features, data, folds, p_work_list, cost, gamma)
    
    print "probabilities vector:", probs

    manager = mp.Manager() ##manager creates a new process to handle variables, may not be the best option  
    
    ##setup for paralelization
    share_lock = mp.Lock() ##create lock to update
    score_lock = mp.Lock()
    work_lock = mp.Lock()
    global_info = manager.Namespace() ##manager for variables
    global_info.best_score = 0.0 ##saves the best score found so far
    global_info.needing_work = []  ##saves the processes that currently need work
    global_info.chosen_to_send = -1  ## saves the selected process that will send the work
    #global_info.last_update = 0.0 ##timestamp of the last change of best_score
    comb_memory = manager.dict()
    
    pipes = []  ##each process needs their own pipe to reveice work
    work_size = [] ##to save the amount of work each process receives
    for i in range(n_proc):
        pipes.append(mp.Pipe())
        work_size.append(50) ##initiate with some work so process don't finish when not receiving work in the beggining
    global_info.work_size = work_size ## saves the amount of work each process has
    
    ##spawn processes
    workers = []  ##to keep track of the workers
    for i in range(1,n_proc):
        p = mp.Process(target=worker_classification, args=(i, p_work_list[i], comb_memory, data, features, folds, settings, pipes, share_lock, global_info, probs, cost, gamma, score_lock, work_lock))
        workers.append(p)
        p.start()
    ##send main process to work
    worker_classification(0, p_work_list[0], comb_memory, data, features, folds, settings, pipes, share_lock, global_info, probs, cost, gamma, score_lock, work_lock)
    ##finish all workers
    for w in workers:
        w.join()
        
    return cost, gamma##result is read from file no just return to main funciton

    
##checks the global info list, in order to find out which is the worker with most work and returns it
def get_worker_with_most_work(global_info, n_proc):
    max = 0
    worker = -1
    for i in range(n_proc):
        if global_info.work_size[i] > max:
            max = global_info.work_size[i]
            worker = i
    return worker
    
##sends an empty list to every pipe so it can terminate
def terminate_workers(id, pipes):
    for i in range(len(pipes)):
        if i == id:
            continue
        _, input = pipes[i]
        input.send([])
    return

##function to ask for work
def ask_for_work(id, global_info, lock, pipes, n_proc, work_lock):
    
    lock.acquire() ##first thing we have to get the lock
    aux = global_info.work_size
    aux[id] = 0
    global_info.work_size = aux
    aux = global_info.needing_work ##add itself to needing work list
    aux.append(id)
    global_info.needing_work = aux
    
    sender = get_worker_with_most_work(global_info, n_proc) ##go find out who is the best sender
    print id, "asking for work, id to send", sender 
    
    if sender == -1: ##there was nobody to send because they all ran out of work
        print id, " terminationg the workers"
        terminate_workers(id, pipes)
        lock.release()
        return []
    global_info.chosen_to_send = sender
    lock.release() ##release the lock
    output, _ = pipes[id] 
    msg = output.recv() ##wait until a essage is received
    update_my_work(id, msg, global_info, work_lock) ##updates the amount of work on the list
    print id, " received work with size ", len(msg)
    return msg
        
        
##divides work amoung n processes, returns lists of working list for every process that needs work and the remaning list to the sharer
def divide_work(work_list, n_workers_needing_work):
    share_lists = []
    for i in range(0, n_workers_needing_work + 1):
        share_lists.append([])
        
    send_to = 0
    for w in work_list:
        share_lists[send_to].append(w)
        send_to += 1
        send_to = send_to % (n_workers_needing_work + 1)
    
    return share_lists[:len(share_lists)-1], share_lists[len(share_lists)-1]


##send work to every process that requested it
def send_work(id, work_list, global_info, pipes, lock, work_lock):
    lock.acquire()
    n_workers_needing_work = len(global_info.needing_work)
    if len(work_list) < n_workers_needing_work * 2: ##if we dont have enough work
        lock.release()
        return work_list, 1
    print id, "Sending work" ## process has enough work to share
    share_lists, work_list = divide_work(work_list, n_workers_needing_work)
    for i in range(n_workers_needing_work):
        _, input = pipes[global_info.needing_work[i]]
        input.send(share_lists[i])
    global_info.needing_work = []
    global_info.chosen_to_send = -1
    update_my_work(id, work_list, global_info, work_lock)
    lock.release()
    return work_list, 0
        
def update_my_work(id, wl, global_info, lock):
    lock.acquire()
    aux = global_info.work_size
    aux[id] = len(wl)
    global_info.work_size = aux
    lock.release()
            
            
##function for each worker to work        
def worker_classification(id, work_list, comb_memory, data, features, folds, settings, pipes, lock, global_info, probs, cost, gamma, score_lock, work_lock):
    n_proc = settings.number_proc
    t_work_ask = 0
    worker_start = time.time()
    mec_cut_nodes = 0 ##number of nodes removed by the second phase mechanism
    #filename = "outputs/out" + str(id) + ".txt" ## to save outputs in case its needed
    best_sets = {}
    update_rate = 10
    count = update_rate
    number_of_tests = 0
    tested_sets = {}  ##to save own tests on process
    #info = []
    rts = []
    depth = False
    last_update = 0.0 ##last time it updated the best global accuracy
    wasted_time = 0.0 ##debug to count wasted time in exchanging work
    send_time = 0.0
    times_work_not_sent = 0
    
    while (True):
        rt = time.time() ##to check how many time it takes to process a subset
        ##if process has no work
        if work_list ==  []: ##ask for work
            t_work_ask += 1 ## to count how many times this process requested work
            ask_time = time.time() ## measure time to ask
            #dont let processes ask for work right after start, make them wait x seconds in order to other processes have time to generate enough work to share
            if number_of_tests < 25: ## the last process when generates their expansions all are repeated and it runs out of work quickly so we have to make him wait before testing
                time.sleep(4)
            work_list = ask_for_work(id, global_info, lock, pipes, n_proc, work_lock)
            aux_time = time.time() - ask_time
            aux_time = round(aux_time,2)
            wasted_time += aux_time
            if work_list == []: ##if it received no work
                break
            
        ##gather a subset to test and remove it from work
        test_subset = work_list[len(work_list)-1]  ##get the subset to test
        del(work_list[len(work_list)-1]) ##delete the subset from the work list ##according to sources removing from the end oof the list is much faster than removing from the end
        
        if depth:
            work_list, last_update, aux_cut = check_cutting(id, last_update, global_info, work_list, settings.search_cutting_time)
            mec_cut_nodes += aux_cut
        else:   ##switch to depth first search and activate the sampling
            if len(test_subset.features) > settings.change_search_size: ##switch stages search
                print "AT_SWITCH:" + str(number_of_tests) + "," + str(len(work_list))
                last_update = time.time() ##time to start measure the updates
                depth = True
            
        
        ##classify subset
        score = classification_part.classify(folds, data, test_subset.features, cost, gamma, settings.svm_kernel)
        
       # info.append((test_subset.features, score)) ##to save all results on count
        test_subset.parents_scores.append(score)
        
        number_of_tests += 1 ##increase number of tests
        
        
        if checkExpand(test_subset, global_info, depth, settings): ##if it's worth expand the node
            work_list = expand_node(id, work_list, comb_memory, test_subset, features, n_proc, tested_sets, depth, probs, settings.estimate_probs) ##expand the node
        
        last_update = update_score(global_info, score, test_subset, best_sets, score_lock, last_update) ##update the top scores
        rts.append(time.time() - rt)
        
        ##update global information 
        count -= 1
        
        ##if process has chosen to send
        if global_info.chosen_to_send == id: ##i have been chosen to send work to someone
            stime = time.time()
            work_list, aux = send_work(id, work_list, global_info, pipes, lock, work_lock)
            times_work_not_sent += aux
            aux_time = time.time() - stime
            aux_time = round(aux_time,2)
            send_time += aux_time 
            count = update_rate ##to update the work list
        
        
        if count < 0:
            ##update global size of work for the process
            ##this is used to give some feedback to the user from time to time, also to globally update the amount of the work of the process
            
            count = update_rate##number of tests untill output again
            update_my_work(id, work_list, global_info, work_lock) 
            
            ##debug info
            sum = 0
            for r in rts:
                sum += r
            avg = sum / float(len(rts))
            avg = round(avg, 4)
            print id, ",", avg, "," , max(best_sets), ",", int(time.time() - worker_start), "," , len(work_list) , ",", str(best_sets[max(best_sets)].features)  
            rts = []
            
            
    total_working_time = time.time() - worker_start
    
    #file_write = open(filename, "a")
    #for t in debug_data:
     #   file_write.write(str(debug_data[t][0]) + "," + str(debug_data[t][1]) + "," + str(debug_data[t][2]) + "\n")
    #file_write.close()
    lock.acquire()
    out_file = open("res.csv", "a")
    out_file.write("PROCESS" + str(id) + "," + str(total_working_time) + "," + str(number_of_tests) + "," + str(max(best_sets)) + "," +str(mec_cut_nodes) + "," + str(wasted_time) + "," + str(send_time) + "," + str(t_work_ask) + ","+ str(times_work_not_sent) + "\n")
    
    for score in best_sets:
        set_info = ""
        for ft in best_sets[score].features:
            set_info += str(ft) + ","
        set_info += str(score) + "\n"
        out_file.write(set_info)
    
    out_file.close()
    lock.release()
     #   aux += "PROCESS " + str(id) + ": " + str(best_sets[score].features) + " -> " + str(score) + " \n"
    #print aux

##check if its time to cut something
def check_cutting(id, last_update, global_info, work_list, cutting_time):
    if ( time.time() - last_update) > cutting_time: ##its time to sample
        if len(work_list) < 100: ##dont remove when the amount of work is not more than 100
            return work_list, last_update, 0
        max_global_score = global_info.best_score ##best score globally found    
        #print "id", id, " cutting of ", len(work_list)
        start_length = len(work_list)
        new_work_list = options_wrapper.sampling(work_list, max_global_score)
        #print "id", id, " got ", len(new_work_list)
        return new_work_list, time.time(), start_length - len(new_work_list)
        
    return work_list, last_update, 0
        
    
##updates the local top of the process and also in case its needed also updates the global score
def update_score(global_info, score, subset, best_sets, lock, last_update):
    if len(best_sets) < 5: ##if we dont have 5 elements yet, lets keep adding
        best_sets[score] = subset
        return last_update
    min_value = min(best_sets)
    if score > min_value:
        if score > global_info.best_score: ##found an global_best that is better than the previous one
            last_update = time.time() ##update the time since we last found something usefull
            
            lock.acquire() #require lock to change the best_score
            if score > global_info.best_score: ##to make sure the best score didn't change while we weere waiting for the lock
                global_info.best_score = score
                global_info.last_update = time.time() ##change the timestamp of the best one found
            lock.release()
        del best_sets[min_value]
        best_sets[score] = subset
    return last_update

##checks the probability of a feature and randomly choses it or not
def use_ft(prob):
    r = random.randint(0,100)
    random_prob = r / 100.0 
    if random_prob <= prob:
        return True
    return False
    
    
##expands a subset
def expand_node(id, work_list, comb_memory, test_subset, features, n_proc, tested_sets, depth, probs, use_probabilities):
    combination = test_subset.features
    new_additions = []
    for ft in features: ##iterate through all possible features
        if ft not in combination:
            if use_probabilities: ## if we are using the probabilities feature
                if not use_ft(probs[ft]): ##if the ft was not selected just proceed to the next one
                    continue
            aux = combination + [ft]
            aux.sort()
            cannonical_name = ','.join(str(e) for e in aux)
            if cannonical_name not in tested_sets: ##maybe a new comb lets add to check it later
                new_additions.append((aux, cannonical_name)) ##at this stage we know that this process didnt try this comb
    new_additions = valid_new_combs(id, new_additions, comb_memory, n_proc)  ##eliminate combinations tested by other process
    aux_list = []
    for new_comb in new_additions: ## add the new additions to the list
        new_list = list(test_subset.parents_scores)  ## too avoid passing a pointer to the fathers list
        aux_list.append(ML_subset(new_comb[0], new_list))
        tested_sets[new_comb[1]] = True
        
    ##if we reached a certain depth lets start expanding on depth
    if not depth:
        work_list = aux_list + work_list
    else:
        work_list = work_list + aux_list
    return work_list

##cut new combinations against the all tests list
def valid_new_combs(id, new_adds, comb_memory, n_proc):
    new_combs = []
    rep_info = comb_memory
    for comb in new_adds:
        if comb[1] not in rep_info:
            new_combs.append(comb)
            rep_info[comb[1]] = True
    
    comb_memory = rep_info
    return new_combs
    

##checks if it worth expand a node based on the previous relations
def checkExpand(subset, global_info, depth, settings):
    scores = subset.parents_scores
    last_score = scores[len(scores)-1]  
    
    if depth: ##if second stage of search
        if last_score >= scores[len(scores)-2]: ## expand while child nodes are improving or at least having the same score as the parents
            return True
    else: ##if on the first stage of the search
        if (last_score + settings.search_threshold) > global_info.best_score: ##expand while the node's score is near the globally best one
            return True
    return False
        



            
                
            
        
        
    
    
    

    

    
    


