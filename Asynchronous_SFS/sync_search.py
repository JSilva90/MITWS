import multiprocessing as mp
import utils
import time
import pandas as pd
import numpy as np
from ML_test import ML_test 
from ML_data import ML_instance


def worker_behaviour(id, work, ml_instance, com):
    master_channel, worker_channel = com
    wasted_time = 0.0
    total_time = time.time()
    task_time = []
    total_wasted_time = []
    while work != []: ##the process works until it doesnt receive more work
        results = []
        print id, " starting round ", len(work[0]), " first ", work[0], " size ", len(work)
        for subset in work: ##test subsets for this round
            #print id, " testing:", subset 
            t_time = time.time()
            score = ml_instance.rf_evaluator(subset)
            #score = ml_instance.svm_evaluator(subset)
            #score = ml_instance.mlp_evaluator(subset)
            #print "testing time: ", round(time.time() - r,1) 
            #s = [str(x) for x in subset]
            #results.append((score, ",".join(s)))
            results.append((score, subset))
            task_time.append(time.time() - t_time)
        worker_channel.send(results)
        results = []
        communicated = False
        print id, " end round processing, wiating for work, average task time", round(sum(task_time)/len(task_time), 2)
        t = time.time()
        while not communicated: ##stall untill receives work from master
            if worker_channel.poll(): ##received com 
                work = worker_channel.recv()
                communicated = True
            time.sleep(1)
        w_t = round(time.time() - t, 2)
        wasted_time += w_t
        total_wasted_time.append(w_t)
        print str(id) + ", " + str(round(time.time()-total_time,2)) + "," + str(round(wasted_time, 2)) + "," + str(round(wasted_time / total_time,4))
    print str(id) + " ended, total_ wasted time: ", wasted_time, " std = ", np.std(task_time)
    df = pd.DataFrame()
    df["WastedTime" + str(id)] = total_wasted_time
    df.to_csv("wasted" + str(id) + ".csv", index=False)

    
def master_behaviour(work, ml_instance, n_workers):
    ##initiate workers
    work_division =  utils.divide_list(n_workers, work)
    st = time.time()
    pipes = []
    workers = []
    work_counter = 0
    for i in range (n_workers):
        c = mp.Pipe()
        pipes.append(c)
        print "master sending to ", i, " size of work ", len(work_division[i]) 
        p = mp.Process(target=worker_behaviour, args=(i, work_division[i], ml_instance, c)) ##cant pass arguments to evaluator
        workers.append(p)
        p.start()
    
    tester = ML_test()
    while work_division[0] != []: ##synchroniously send work until search is over
        
        i = 0
        print "Master waiting for work"
        for master_channel, worker_channel in pipes:
            
            #master_channel, worker_channel = pipe
            worker_results = master_channel.recv() #3receive results
            #print "Received work from " + str(i)
            i += 1
            tester.round_results += worker_results
        tester = ml_instance.generator(tester)
        print "Generated ", len(tester.round_subsets), " subsets, already explored: ", len(tester.history.keys())
        work_counter += len(tester.round_subsets)
        work_division = utils.divide_list(n_workers, tester.round_subsets)
        for i in range(n_workers):
            master_channel, worker_channel = pipes[i]
            master_channel.send(work_division[i])
            #print "work_sent to " + str(i)
    print "Ending Workers"
    for w in workers:
        w.join()
    print "end ", round(time.time()-st,2)
    print "total counter ", work_counter
    print "total explored ", len(tester.history)
    tester.save_history(filename="history_sync.csv")
    tester.save_scores(filename="scores_sync.csv")
    tester.save_expanded(filename="expanded_sync.csv")
    return tester
        
        
    
    
    
    
    

    
