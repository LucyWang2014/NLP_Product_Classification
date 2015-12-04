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


log.info('importing modules...')
import pandas as pd
import numpy as np
import pdb
import cPickle as pkl
import image_processing
#import bag_of_words


#TODO: 
# 1. test and debug train_val_split
# 2. run image feature extraction on shuffled dataset.  We aren't going to have time to get them all.  

def get_brand_index(trainDF,testDF):
    
    def apply_brand_index(brand,brand_list):
        if brand in brand_list:
            return brand_list.index(brand)
        else:
            return None
        
    brands = list(trainDF.brand.unique())
    trainDF['brand_num']=trainDF.brand.apply(apply_brand_index,args=[brands])
    testDF['brand_num']=testDF.brand.apply(apply_brand_index,args=[brands])

def shuffle_and_downsample(df,samples):
    '''
    shuffle dataframe, including previous indexes, then downsample
    args:
        samples: number of samples
    '''
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

def X_y_split(df,y_label):
    X=df.loc[:,'brand_num','description_clean','large_image_URL','cat_1','cat_2','cat_3','cat_1_num','cat_2_num','cat_3_num']
    y=df.loc[:,y_label]
    return X,y

def merge_data(sets,csv_path,text_path, image_path):
    '''
    merge together the datasets to be used in the model
    args:
        sets: list of datasets to be used
    '''
    #TODO

def main(datadir,
        trainpath,
        testpath,
        train_samples,
        test_samples,
        val_portion,
        y_label,
        use_images,
        use_text):
    '''
    1. run train_val_split on training
    1b. run shuffle on test
    2. if text:
        a. train tokenizer
        b. convert text data to bag of words matrix
    3. if images:
        a. extract image data
    4. merge datasets

    args:
        train_df: training dataframe
        test_df: test dataframe
        use_images: boolean.  want to use image data?
        use_text: boolean. want to use text data?
    '''

    trainDF = pd.read_csv(trainpath,header = 0, index_col = 0,low_memory = False)
    testDF = pd.read_csv(testpath,header = 0, index_col = 0,low_memory = False)

    testDF = shuffle_and_downsample(testDF,test_samples)
    trainDF = shuffle_and_downsample(trainDF,train_samples)
    trainDF,valDF = train_val_split(trainDF,val_portion)

    return trainDF, valDF, testDF
    #if use_images:    

    #if use_text:


    #X_trainDF = X_y_split(df,y_label)
    #data = (X_train, y_train, X_val, y_val, X_test, y_test)
    #return data

if __name__ == '__main__':
    home = os.path.join(os.path.dirname(__file__),'..')
    #local datadir
    #datadir = os.path.join(home,'data') + '/'

    #hpc datadir
    datadir = '/scratch/cdg356/spring/data/'

    trainpath = datadir + 'train_set.csv'
    testpath = datadir + 'test_set.csv'
    train_samples=1000
    test_samples=1000
    val_portion=0.1
    y_label='cat_1_num'
    use_images=False
    use_text=False
    
    trainDF, valDF, testDF = main(datadir,
                                trainpath,
                                testpath,
                                train_samples,
                                test_samples,
                                val_portion,
                                y_label,
                                use_images,
                                use_text)

    dataset="train"
    iloc0=0
    iloc1=200
    save_freq=100
    out_pickle_name=dataset+'_image_features'

    image_processing.get_selected_image_features(trainDF,
                                datadir,
                                dataset,
                                iloc0,
                                iloc1,
                                save_freq,
                                out_pickle_name,
                                width=224,
                                filetype='jpg')
