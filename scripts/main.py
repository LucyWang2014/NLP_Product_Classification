'''
main.py
End-to-end script for running all processes.  
'''
__author__='Charlie Guthrie'

from utils import create_log,plog
create_log(__file__)

plog('importing main.py modules...')
import os
import data_prep
import models
import pdb

home = os.path.join(os.path.dirname(__file__),'..')
#local datadir
datadir = os.path.join(home,'data') + '/'

#hpc datadir
#datadir = '/scratch/cdg356/spring/data/'
data,n_values = data_prep.main(datadir,
                                train_samples=10000,
                                test_samples=1000,
                                val_portion=0.1,
                                use_images=False,
                                use_text=True)

plog("Starting model...")
models.main(data,n_values)
