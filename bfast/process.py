'''
Created on Oct 15, 2019

@author: fgieseke
'''

import multiprocessing

def perform_task_in_parallel(task, params_parallel, n_jobs=1):
    """ Performas a task in parallel
     
    Parameters
    ----------
    task : callable
        The function/procedure that shall be executed
    params_parallel : list
        The parallel parameters
    n_jobs : int, default 1
        The number of jobs that shall be used
    """
          
    if n_jobs == 1:
        
        results = []
        for param in params_parallel:
            r = start_via_single_thread(task, [param])
            results.append(r)
        return results
    
    pool = multiprocessing.Pool(n_jobs)
    results = pool.map(task, params_parallel)

    pool.close()
    pool.join()
        
    return results

def wrapped_task(proc_num, task, args, kwargs, return_dict):
    
    return_dict[proc_num] = task(*args, **kwargs)
        
def start_via_single_thread(task, args, kwargs={}):
    
    manager = multiprocessing.Manager()
    return_dict = manager.dict()

    proc_num = 0
    proc = multiprocessing.Process(target=wrapped_task, args=(proc_num, task, args, kwargs, return_dict))
    proc.daemon = False
    proc.start()
    
    proc.join()
        
    return return_dict[proc_num]