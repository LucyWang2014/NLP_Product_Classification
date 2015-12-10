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
    'train_samples': 10000, #10k, 50k, 100k. test is 10% of train
    'val_portion': 0.1,
    'use_images': True, # T, F
    'use_text': False, # T, F
    'train_image_fn': 'train_image_features_0_67500.pkl',
    'test_image_fn': 'test_image_features_0_100000.pkl',
    'debug': True,

    #MODEL PARAMS,
    'num_epochs': 200, #200
    'depth': 3, #3
    'width': 256,
    'drop_in': .2,
    'drop_hid': .5,
    'batch_size': 256, 
    'learning_rate': 0.01,
    'valid_freq': 1000, #1000
    'reload_model': None,
    'num_targets': 3
}

#Convert params dictionary to actual variables. Magic!
locals().update(options_dict)

#Define result path
start_time = datetime.now()
log_time = start_time.strftime('%Y%m%d_%H%M%S')
if not os.path.exists('../results/'):
    os.makedirs('../results/')
results_path = '../results/model_results_%s_%s_%s.npz' %(use_images,use_text,log_time)

test_samples = int(0.1*train_samples)
data,n_values = data_prep.main(datadir,
                                train_samples,
                                test_samples,
                                val_portion,
                                use_images,
                                use_text,
                                train_image_fn,
                                test_image_fn,
                                debug)

plog("Starting model...")

params, preds = models.train_simple_model(data,
        n_values,
        num_epochs,
        depth,
        width,
        drop_in,
        drop_hid,
        batch_size,
        learning_rate,
        valid_freq,
        results_path,
        options_dict,
        reload_model,
        num_targets)
