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

import pandas as pd
import cPickle as pkl
import os

def get_indexes(fname):
    iloc0 = fname.split('.')[0].split('_')[-2]
    iloc1 = fname.split('.')[0].split('_')[-1]
    return int(iloc0),int(iloc1)

def stitch_files(basename):
    #datadir = '../data/'
    datadir = '/scratch/cdg356/spring/data/'
    featuredir = datadir+basename+'/'
    
    #Get list of indexes
    iloc0_list = []
    iloc1_list = []
    for root, dirs, files in os.walk(featuredir):
        for fname in files:
            idx_range = get_indexes(fname)
            iloc0_list.append(idx_range[0])
            iloc1_list.append(idx_range[1])
    iloc0_list.sort()
    iloc1_list.sort()
    
    #Load files
    for i,iloc0 in enumerate(iloc0_list):
        iloc1=iloc1_list[i]
        fname = basename + "_%i_%i.pkl" %(iloc0,iloc1)
        plog("loading %s..." %fname)
        with open(featuredir + fname,'rb') as f:
            if i==0:
                df=pkl.load(f)
            else:
                df2=pkl.load(f)
                df=pd.concat([df,df2])
        plog("df shape: %s" %str(df.shape))
    max_index = max(iloc1_list)
    outname = datadir + basename + '_0_%s.pkl'%max_index

    plog("writing to file...")
    with open(outname,'wb') as f:
        pkl.dump(df,f)
        

if __name__ == '__main__':
    stitch_files('test_image_features')
