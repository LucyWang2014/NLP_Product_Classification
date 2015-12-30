'''
main.py
End-to-end script for running all processes.  
'''
__author__='Charlie Guthrie'

from utils import create_log,plog,fplog
create_log(__file__)
import sys

#Command-line arguments
if len(sys.argv)<2:
    plog("Usage: python main.py [num_train_samples] [use_images|use_text]")
    sys.exit()
else:
    train_samples = int(sys.argv[1])
    if 'use_images' in sys.argv:
        use_images=True
    else:
        use_images=False
    if 'use_text' in sys.argv:
        use_text=True
    else:
        use_text=False

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
    'train_samples': train_samples, #10k, 50k, 100k. test is 10% of train
    'val_portion': 0.1,
    'use_images': use_images, # T, F
    'use_text': use_text, # T, F
    'train_image_fn': 'train_image_features_0_100000.pkl',
    'test_image_fn': 'test_image_features_0_100000.pkl',
    'debug': False,

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

if use_images:
    image_str='images_'
else:
    image_str=''

if use_text:
    text_str='text_'
else:
    text_str=''

sample_str = str(train_samples/1000)+'k_'
results_path = '../results/results_%s%s%s%s.npz' %(sample_str,image_str,text_str,log_time)

print train_samples
test_samples = int(0.1*train_samples)

dstart=datetime.now()
plog("Checking to see if prepped data already available...")
#TODO: change this to check for the appropriate train, val, test sets
data_path = datadir + 'model_data_%i_%r_%s_%s.pkl'%(train_samples,val_portion,use_images,use_text)
if os.path.exists(outpath):
    plog("Data found. Moving on to models.py")
else:
    plog("Prepped data not available.")
    plog("Starting data_prep with %s training samples; use_images=%s; use_text=%s" %(train_samples,use_images,use_text))
    data_prep.main(datadir,
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
