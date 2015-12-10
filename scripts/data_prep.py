'''
data_prep.py

Starting with the csv's, ending with X_train, y_train, X_val, y_val, X_test, y_test
Where X's are feature vectors and y's are classifier integers
'''
__author__='Charlie Guthrie'

from utils import create_log,plog
create_log(__file__)

plog('importing modules...')
from datetime import datetime
import os
import pandas as pd
import numpy as np
import pdb
import cPickle as pkl
import image_processing
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
    '''
    load images from pkl files and convert to matrices
    '''

    plog("Loading train image features from %s..." %train_imagepath)
    with open(train_imagepath,'rb') as f:
        imageDF=pkl.load(f)
    
    if test_imagepath is not None:
        plog("Loading test image features from %s..." %test_imagepath)
        with open(test_imagepath,'rb') as f:
            test_imageDF = pkl.load(f)

        test_image_matrix = test_imageDF.as_matrix()
        test_image_matrix = test_image_matrix[:testDF.shape[0],:]
        assert test_image_matrix.shape[0]==testDF.shape[0]
    else: test_image_matrix=None

    image_matrix = imageDF.as_matrix()
    train_image_matrix = image_matrix[:trainDF.shape[0],:]
    val_image_matrix = image_matrix[trainDF.shape[0]:trainDF.shape[0] + valDF.shape[0],:]

    return (train_image_matrix, val_image_matrix, test_image_matrix)


def X_y_split(df):
    '''
    Split data frame into matrices of X and y variables, then output as float32/int32
    '''
    #X=df.loc[:,'brand_num','description_clean','large_image_URL','cat_1','cat_2','cat_3','cat_1_num','cat_2_num','cat_3_num']
    X = df.loc[:,'brand_num'].values
    X = np.reshape(X,(len(X),1)).astype(np.float32)
    y1 = df.loc[:,'cat_1_num'].values.astype(np.int32)
    y2 = df.loc[:,'cat_2_num'].values.astype(np.int32)
    y3 = df.loc[:,'cat_3_num'].values.astype(np.int32)
    return X,y1,y2,y3



def conditional_hstack(other,bow,image,dataset_name):
    '''
    assumes other is present.
    if bag of words is not none, hstack it
    if image is not None, hstack it
    '''
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
    returns: 2D float32 numpyarrays
    '''
    #HACK: splitting None into 3
    if bows is None:
        bows = (None,None,None)
    if images is None:
        images= (None,None,None)

    plog("Merging data...")
    X_train = conditional_hstack(others[0],bows[0],images[0],'train')
    X_val = conditional_hstack(others[1],bows[1],images[1],'val')
    X_test = conditional_hstack(others[2],bows[2],images[2],'test')

    return X_train.astype(np.float32), X_val.astype(np.float32), X_test.astype(np.float32)
    

def prepDFs(datadir,
        train_samples=10000,
        test_samples=1000,
        val_portion=0.1,
        debug=False):
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
    if(debug):
        trainpath = datadir + 'head_train_set.csv'
        testpath = datadir + 'head_test_set.csv'
        train_samples = 90
        test_samples = 90
    else:
        trainpath = datadir + 'train_set.csv'
        testpath = datadir + 'test_set.csv'

    plog("Loading train csv...")
    trainDF = pd.read_csv(trainpath,header = 0, index_col = 0,low_memory = False)
    plog("Loading test csv...")
    testDF = pd.read_csv(testpath,header = 0, index_col = 0,low_memory = False)

    brand_list = get_brand_index(trainDF,testDF)
    with open(datadir + 'brand_list.pkl','wb') as f:
        pkl.dump(brand_list,f)

    trainDF = shuffle_and_downsample(trainDF,train_samples)
    testDF = shuffle_and_downsample(testDF,test_samples)
    return trainDF,testDF

def main(datadir,
        train_samples=10000,
        test_samples=1000,
        val_portion=0.1,
        use_images=True,
        use_text=True,
        train_image_fn='train_image_features_0_2500.pkl',
        test_image_fn='test_image_features_0_2500.pkl',
        debug=False):
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


    if(debug):
        trainpath = datadir + 'head_train_set.csv'
        testpath = datadir + 'head_test_set.csv'
        train_imagepath = datadir + 'train_image_features_0_2500.pkl'
        test_imagepath = datadir + 'test_image_features_0_2500.pkl'
        train_samples = 90
        test_samples = 90
    else:
        trainpath = datadir + 'train_set.csv'
        testpath = datadir + 'test_set.csv'
        train_imagepath = datadir + train_image_fn
        test_imagepath = datadir + test_image_fn

    dstart=datetime.now()
    plog("Checking to see if prepped data already available...")
    outpath = datadir + 'model_data_%i_%r_%s_%s.pkl'%(train_samples,val_portion,use_images,use_text)
    if os.path.exists(outpath):
        plog("Data found.  Loading...")
        with open(outpath,'rb') as f:
            data,n_values = pkl.load(f)

        dfin = datetime.now()
        plog("Data loading time: %s" %(dfin-dstart))
        return data,n_values

    plog("Prepped data not available.  Preparing data...")


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
        image_data = get_image_matrices(train_imagepath,test_imagepath,trainDF, valDF, testDF)
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

    keys = ['y_1','y_2','y_3']
    values = [max(d)+1 for d in train_data[1:]]
    n_values = dict(zip(keys,values))
    data = (train_data, val_data, test_data)

    plog("Data loaded.  Saving to %s" %outpath)
    with open(outpath,'wb') as f:
        pkl.dump((data,n_values),f)

    dfin = datetime.now()
    plog("Data loading time: %s" %(dfin-dstart))

    return data,n_values

if __name__ == '__main__':
    home = os.path.join(os.path.dirname(__file__),'..')
    datadir = os.path.join(home,'data') + '/'

    trainDF,testDF = prepDFs(datadir,
                            train_samples=None,
                            test_samples=100,
                            val_portion=0.1,
                            debug=False)

    df=trainDF
    dataset="train"
    iloc0=69000
    iloc1=200000
    save_freq=10
    batch_size = 250
    out_pickle_name=dataset+'_image_features/'+dataset+'_image_features'

    image_processing.get_selected_image_features(df,
                                datadir,
                                dataset,
                                iloc0,
                                iloc1,
                                save_freq,
                                out_pickle_name,
                                batch_size,
                                width=224,
                                filetype='jpg')
