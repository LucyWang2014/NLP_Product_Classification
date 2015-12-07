'''
main.py
End-to-end script for running all processes.  
'''
__author__='Charlie Guthrie'

#LOGGING BOILERPLATE
from datetime import datetime
start_time = datetime.now()
import os
#make log file if it doesn't exist
if not os.path.exists('logs'):
    os.makedirs('logs')
#logging boilerplate.  To log anything type log.info('stuff you want to log')
import logging as log
#log file with filename, HMS time
log.basicConfig(filename='logs/%s%s.log' %(__file__.split('.')[0],start_time.strftime('_%Y%m%d_%H%M%S')),level=log.DEBUG)

def plog(msg):
    print msg
    log.info(msg)

plog('importing modules...')
import data_prep
import models
import pdb

home = os.path.join(os.path.dirname(__file__),'..')
#local datadir
#datadir = os.path.join(home,'data') + '/'

#hpc datadir
datadir = '/scratch/cdg356/spring/data/'
train_data, val_data, test_data = data_prep.main(datadir,
                                                train_samples=1000,
                                                test_samples=100,
                                                val_portion=0.1,
                                                use_images=False,
                                                use_text=True)

X_train,y_train,_,_= train_data
X_val,y_val,_,_= val_data
X_test,y_test,_,_ = test_data

data_to_model = (X_train, y_train, X_val, y_val, X_test, y_test)
pdb.set_trace()
plog("Starting model...")
models.main(data_to_model,model='mlp', num_epochs=10)
