'''
main.py
End-to-end script for running all processes.  
'''
__author__='Charlie Guthrie'

from utils import create_log,plog,fplog
create_log(__file__)

plog('importing main.py modules...')
import os
import data_prep
import models
import pdb

home = os.path.join(os.path.dirname(__file__),'..')
datadir = os.path.join(home,'data') + '/'

data,n_values = data_prep.main(datadir,
                                train_samples=67000,
                                test_samples=10000,
                                val_portion=0.1,
                                use_images=False,
                                use_text=False,
                                debug=False)

plog("Starting model...")

params, preds = models.train_simple_model(data = data,
        n_values = n_values,
        num_epochs=5,
        depth = 10,
        width = 256,
        batch_size = 256,
        learning_rate = 0.01,
        valid_freq = 100,
        save_path = '../results/simple_mlp/',
        saveto = 'simple_mlp.npz',
        reload_model = None,
        num_targets = 3)
