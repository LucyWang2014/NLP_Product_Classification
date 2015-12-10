#LOGGING BOILERPLATE
from datetime import datetime
import os
import logging as log
import cPickle as pkl

#logging boilerplate.  To log anything type log.info('stuff you want to log')

def create_log(script_name):
    start_time = datetime.now()
    #make log file if it doesn't exist
    if not os.path.exists('../logs'):
        os.makedirs('../logs')

    #log file with filename, HMS time
    log_time = start_time.strftime('_%Y%m%d_%H%M%S')
    log_fname = '../logs/%s%s.log' %(script_name.split('.')[0],log_time)
    log.basicConfig(filename=log_fname,level=log.DEBUG)


def plog(msg):
    print msg
    log.info(msg)

def fplog(msg):
    print(msg)
    log.info(msg)

def create_results_file(param_dict):
    start_time = datetime.now()
    log_time = start_time.strftime('%Y%m%d_%H%M%S')
    if not os.path.exists('../results/'):
        os.makedirs('../results/')
    results_path = '../results/model_resuts_%s.pkl' %log_time
    with open(results_path,'wb') as f:
        pkl.dump(param_dict,f)
    return results_path

def save_to_results_file(var_string,var,results_path):
    '''
    loads parameter dictionary and adds variable string and variable value
    '''
    with open(results_path,'r+') as f:
        param_dict = pkl.load(f)
    param_dict.update({var_string:var})
    with open(results_path,'wb') as f:
        pkl.dump(param_dict,f)