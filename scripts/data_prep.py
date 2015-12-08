'''
data_prep.py

Starting with the csv's, ending with X_train, y_train, X_val, y_val, X_test, y_test
Where X's are feature vectors and y's are classifier integers
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
import pandas as pd
import numpy as np
import pdb
import cPickle as pkl
#import image_processing
import bag_of_words


#TODO: 
# 1. Validation set currently depends on number of samples.  Need to be aware of this with image processing
# 2. run image feature extraction on shuffled dataset.  We aren't going to have time to get them all.  


def get_brand_index(trainDF,testDF):
    '''
    converts brand names to indexes
    '''
    def apply_brand_index(brand,brand_list):
        if brand in brand_list:
            return brand_list.index(brand)
        else:
            return None
        
    brands = list(trainDF.brand.unique())
    trainDF['brand_num']=trainDF.brand.apply(apply_brand_index,args=[brands])
    testDF['brand_num']=testDF.brand.apply(apply_brand_index,args=[brands])
    return brands

def shuffle_and_downsample(df,samples):
    '''
    shuffle dataframe, including previous indexes, then downsample
    args:
        samples: number of samples
    '''
    #random seed 9 makes sure we always get the same shuffle.
    np.random.seed(9)
    assert df.shape[0]>2
    df = df.reindex(np.random.permutation(df.index)).copy()

    if samples is not None:
        assert samples<=df.shape[0]
        sampleDF = df.iloc[:samples,:]
    else:
        sampleDF = df
    return sampleDF

def train_val_split(df,val_portion):
    '''
    split dataframe into validation and training sets
    
    args:
        df: data frame to split
        val_portion: fraction (between 0 and 1) of samples to devote to validation
    returns:
        trainDF
        valDF  
    '''
    assert val_portion<1 and val_portion>0
    val_samples = int(df.shape[0]*val_portion)
    train_samples = df.shape[0] - val_samples
    trainDF = df.iloc[:train_samples,:]
    valDF = df.iloc[train_samples:train_samples+val_samples,:]

    assert valDF.shape[0] + trainDF.shape[0] == df.shape[0]
    assert valDF.shape[1]==trainDF.shape[1]
    return trainDF, valDF

#Get text data
def build_text_matrices(datadir, tokenizer_path, trainDF, valDF, testDF):
    plog("Building text matrices...")
    with open(tokenizer_path) as f:
        tokenizer=pkl.load(f)
    train_text_matrix_path=datadir + 'train_text.pkl'
    val_text_matrix_path=datadir + 'val_text.pkl'
    test_text_matrix_path=datadir + 'test_text.pkl'

    bow_train, idx_train = bag_of_words.series_to_bag_of_words(trainDF.description_clean,tokenizer,train_text_matrix_path,mode="binary")
    bow_val, idx_val = bag_of_words.series_to_bag_of_words(valDF.description_clean,tokenizer,val_text_matrix_path,mode="binary")
    bow_test, idx_test = bag_of_words.series_to_bag_of_words(testDF.description_clean,tokenizer,test_text_matrix_path,mode="binary")

    plog("bow_train type: %s" %type(bow_train))
    return (bow_train, bow_val, bow_test),(idx_train,idx_val,idx_test)

def get_image_matrices(train_imagepath,test_imagepath, trainDF, valDF, testDF):
    plog("Getting image matrices...")

    def df2matrix(df):
        return np.array([x[0,:] for x in df.iloc[:,0]]).astype(np.float32)

    with open(train_imagepath,'rb') as f:
        imageDF=pkl.load(f)

    if test_imagepath is not None:
        with open(test_imagepath,'rb') as f:
            test_imageDF = pkl.load(f)

        test_image_matrix = df2matrix(test_imageDF)
        test_image_matrix = test_image_matrix[:testDF.shape[0]]
        assert test_image_matrix.shape[0]==testDF.shape[0]
    else: test_image=None



    image_matrix = df2matrix(imageDF)
    train_image_matrix = image_matrix[:trainDF.shape[0],:]
    val_image_matrix = image_matrix[trainDF.shape[0]:trainDF.shape[0] + valDF.shape[0],:]

    return (train_image_matrix, val_image_matrix, test_image)


def X_y_split(df):
    #X=df.loc[:,'brand_num','description_clean','large_image_URL','cat_1','cat_2','cat_3','cat_1_num','cat_2_num','cat_3_num']
    X = df.loc[:,'brand_num'].values
    X = np.reshape(X,(len(X),1))
    y1 = df.loc[:,'cat_1_num'].values
    y2 = df.loc[:,'cat_2_num'].values
    y3 = df.loc[:,'cat_3_num'].values
    return X,y1,y2,y3

def conditional_hstack(other,bow,image,dataset_name):
    if other is not None:
        X=other
        if bow is not None:
            assert bow.shape[0]==X.shape[0]
            X = np.hstack((X,bow))
        else:
            plog("Bag of words data missing from %s" %dataset_name)
        if image is not None:
            assert image.shape[0]==X.shape[0]
            X = np.hstack((X,image))
        else:
            plog("Image data missing from %s" %dataset_name)
    return X

def merge_data(bows,images,others):
    '''
    merge together the datasets to be used in the model
    args:
        sets: list of datasets to be used
    '''
    #HACK: splitting None into 3
    if bows is None:
        bows = (None,None,None)
    if images is None:
        images= (None,None,None)

    #plog("Merging data...")
    X_train = conditional_hstack(others[0],bows[0],images[0],'train')
    X_val = conditional_hstack(others[1],bows[1],images[1],'val')
    X_test = conditional_hstack(others[2],bows[2],images[2],'test')
    return X_train,X_val,X_test
    

def prepDFs(datadir,
        train_samples=10000,
        test_samples=1000,
        val_portion=0.1,
        use_images=True,
        use_text=True):
    '''
    1. run train_val_split on training
    1b. run shuffle on test
    2. if text:
        a. train tokenizer
        b. convert text data to bag of words matrix
    3. if images:
        a. extract image data
    4. merge datasets

    returns: X_train,y_train,X_val,y_val,X_test,y_test
    '''

    trainpath = datadir + 'train_set.csv'
    testpath = datadir + 'test_set.csv'
    train_imagepath = datadir + 'train_image_features_0_10000.pkl'

    plog("Loading train csv...")
    trainDF = pd.read_csv(trainpath,header = 0, index_col = 0,low_memory = False)
    plog("Loading test csv...")
    testDF = pd.read_csv(testpath,header = 0, index_col = 0,low_memory = False)

    brand_list = get_brand_index(trainDF,testDF)
    with open(datadir + 'brand_list.pkl','wb') as f:
        pkl.dump(brand_list,f)

    trainDF = shuffle_and_downsample(trainDF,train_samples)
    trainDF,valDF = train_val_split(trainDF,val_portion)
    testDF = shuffle_and_downsample(testDF,test_samples)
    return trainDF,valDF,testDF

def main(datadir,
        train_samples=10000,
        test_samples=1000,
        val_portion=0.1,
        use_images=True,
        use_text=True):
    '''
    1. run train_val_split on training
    1b. run shuffle on test
    2. if text:
        a. train tokenizer
        b. convert text data to bag of words matrix
    3. if images:
        a. extract image data
    4. merge datasets

    returns: X_train,y_train,X_val,y_val,X_test,y_test
    '''

    trainpath = datadir + 'train_set.csv'
    testpath = datadir + 'test_set.csv'
    train_imagepath = datadir + 'train_image_features_0_10000.pkl'

    plog("Loading train csv...")
    trainDF = pd.read_csv(trainpath,header = 0, index_col = 0,low_memory = False)
    plog("Loading test csv...")
    testDF = pd.read_csv(testpath,header = 0, index_col = 0,low_memory = False)

    brand_list = get_brand_index(trainDF,testDF)
    with open(datadir + 'brand_list.pkl','wb') as f:
        pkl.dump(brand_list,f)

    trainDF = shuffle_and_downsample(trainDF,train_samples)
    trainDF,valDF = train_val_split(trainDF,val_portion)
    testDF = shuffle_and_downsample(testDF,test_samples)
    #Load text data
    t0 = datetime.now()
    if use_text:
        bow_data,idxs=build_text_matrices(datadir, 'tokenizer_5000.pkl', trainDF, valDF, testDF)
        t1 = datetime.now()
        plog("Time to load text: %s" %str(t1-t0))
    else:
        bow_data=None

    #Load image data
    t1 = datetime.now()
    if use_images:
        image_data = get_image_matrices(train_imagepath,None,trainDF, valDF, testDF)
        t2 = datetime.now()
        plog("Time to load images: %s" %str(t2-t1))
    else:
        image_data=None

    #Load other data
    other_train,y1_train,y2_train,y3_train=X_y_split(trainDF)
    other_val,y1_val,y2_val,y3_val=X_y_split(valDF)
    other_test,y1_test,y2_test,y3_test=X_y_split(testDF)
    other_data = (other_train,other_val,other_test)

    X_train,X_val,X_test = merge_data(bow_data,image_data,other_data)

    train_data = X_train, y1_train, y2_train, y3_train
    val_data = X_val, y1_val, y2_val, y3_val
    test_data = X_test, y1_test, y2_test, y3_test
    return train_data, val_data, test_data

if __name__ == '__main__':
    home = os.path.join(os.path.dirname(__file__),'..')
    #local datadir
    #datadir = os.path.join(home,'data') + '/'

    #hpc datadir
    datadir = '/scratch/cdg356/spring/data/'
    trainDF,valDF,testDF = data_prep.prepDFs(datadir,
                                                train_samples=10000,
                                                test_samples=1000,
                                                val_portion=0.1,
                                                use_images=True,
                                                use_text=True)

    

    df=trainDF
    dataset="train"
    iloc0=30000
    iloc1=100000
    save_freq=1000
    out_pickle_name=dataset+'_image_features/'+dataset+'_image_features'

    image_processing.get_selected_image_features(df,
                                datadir,
                                dataset,
                                iloc0,
                                iloc1,
                                save_freq,
                                out_pickle_name,
                                width=224,
                                filetype='jpg')
