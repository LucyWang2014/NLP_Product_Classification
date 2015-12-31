'''
Stitch separate image files together
'''
__author__='Charlie Guthrie'
from utils import create_log,plog
create_log(__file__)
import pandas as pd
import cPickle as pkl
import os

def get_indexes(fname):
    '''
    read filename and return indexes embedded therein
    '''
    iloc0 = fname.split('.')[0].split('_')[-2]
    iloc1 = fname.split('.')[0].split('_')[-1]
    return int(iloc0),int(iloc1)

def stitch_files(basename,idx_start=0,idx_finish=None):
    '''
    Cycles through all files in the basename directory and stacks them together.
    args:
        basename: name without indexes, e.g. 'train_image_features'
        idx_start: starting index (usually 0)
        idx_finish: last index of the output file
    returns:
        none.  saves pickle of images stitched together
        
    '''
    datadir = '../data/'
    featuredir = datadir+basename+'/'
    
    #Get list of indexes
    iloc0_list = []
    iloc1_list = []
    for root, dirs, files in os.walk(featuredir):
        for fname in files:
            idx_range = get_indexes(fname)
            if idx_range[0] is not None and idx_range[1] is not None:
                if idx_range[0]>=idx_start and idx_range[1]<=idx_finish:
                    iloc0_list.append(idx_range[0])
                    iloc1_list.append(idx_range[1])
    iloc0_list.sort()
    iloc1_list.sort()

    #Make sure there are no duplicates present
    assert len(iloc0_list)==len(set(iloc0_list))
    assert len(iloc1_list)==len(set(iloc1_list))

    #Make sure there are no gaps, i.e. that iloc1 of one file = iloc0 of the next
    for i in range(len(iloc0_list)-1):
        assert iloc0_list[i+1]==iloc1_list[i]

    max_index = max(iloc1_list)

    # A couple sanity checks
    assert max_index == iloc1_list[-1]
    if idx_finish is not None:
        assert idx_finish==max_index
    assert idx_start==iloc0_list[0]

    #Initialize memmap
    shape = (max_index+1,4096)
    shape_str = "_".join(str(i) for i in shape)
    map_name = datadir + basename + '_' + shape_str + '.mm'
    mm = np.memmap(map_name, dtype='float32', mode='w+', shape=shape)

    #Load files
    for i,iloc0 in enumerate(iloc0_list):
        iloc1=iloc1_list[i]
        fname = basename + "_%i_%i.pkl" %(iloc0,iloc1)
        plog("loading %s..." %fname)
        with open(featuredir + fname,'rb') as f:
            df=pkl.load(f)
            mm[iloc0:iloc1-1]=df.values.astype(np.float32)
        plog("df shape: %s" %str(df.shape))
        plog("mm shape: %s" %str(mm[iloc0:iloc1-1].shape))

        

if __name__ == '__main__':
    stitch_files('train_image_features',0,200000)
