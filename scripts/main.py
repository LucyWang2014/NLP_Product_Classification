'''
main.py
End-to-end script for running all processes.  
'''
__author__='Charlie Guthrie'

from utils import create_log,plog,fplog,create_results_file,save_to_results_file
create_log(__file__)

plog('importing main.py modules...')
import os
import data_prep
import models
import pdb
from datetime import datetime

home = os.path.join(os.path.dirname(__file__),'..')
datadir = os.path.join(home,'data') + '/'


#DATA PREP PARAMS
options_dict = {
    'train_samples': 50000, #10k, 50k, 100k. test is 10% of train
    'val_portion': 0.1,
    'use_images': False, # T, F
    'use_text': False, # T, F
    'debug': False,

    #MODEL PARAMS,
    'num_epochs': 5, #200
    'depth': 3, #3
    'width': 256,
    'batch_size': 256, 
    'learning_rate': 0.01,
    'valid_freq': 1000, #1000
    'reload_model': None,
    'num_targets': 3
}

#Define result path
log_time = start_time.strftime('%Y%m%d_%H%M%S')
if not os.path.exists('../results/'):
    os.makedirs('../results/')
results_path = '../results/model_resuts_%s.pkl' %log_time

#Convert params dictionary to actual variables. Magic!
locals().update(param_dict)


data,n_values = data_prep.main(datadir,
                                train_samples,
                                test_samples,
                                val_portion,
                                use_images,
                                use_text,
                                debug)

plog("Starting model...")

params, preds = models.train_simple_model(data = data,
        n_values = n_values,
        num_epochs,
        depth,
        width,
        batch_size,
        learning_rate,
        valid_freq,
        results_path,
        options_dict,
        reload_model,
        num_targets)