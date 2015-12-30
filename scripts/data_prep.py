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
import bag_of_words
from sklearn.preprocessing import OneHotEncoder


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

def get_brand_index(trainDF,valDF,testDF):
    '''
    converts brand names to indexes.  Unknown brands get coded zero
    '''
    def apply_brand_index(brand,brand_list):
        if brand in brand_list:
            return brand_list.index(brand)
        else:
            return 0
        
    brands = list(trainDF.brand.unique())
    #add a zero for unknowns
    brands.insert(0, 'NA')

    trainDF['brand_num']=trainDF.brand.apply(apply_brand_index,args=[brands])
    valDF['brand_num']=valDF.brand.apply(apply_brand_index,args=[brands])
    testDF['brand_num']=testDF.brand.apply(apply_brand_index,args=[brands])
    
    trainDF['brand_num'].fillna(0, inplace=True)
    valDF['brand_num'].fillna(0, inplace=True)
    testDF['brand_num'].fillna(0, inplace=True)
    return brands

def build_brand_matrices(trainDF, valDF, testDF):
    '''
    one-hot encode brand indexes
    '''
    brand_list = get_brand_index(trainDF,valDF,testDF)
    with open(datadir + 'brand_list.pkl','wb') as f:
        pkl.dump(brand_list,f)

    plog("Building brand matrices...")
    enc = OneHotEncoder()
    train_vect = np.reshape(trainDF.brand_num.values,(-1,1))
    brands_train = enc.fit_transform(train_vect).toarray()

    val_vect = np.reshape(valDF.brand_num.values,(-1,1))    
    brands_val = enc.transform(val_vect).toarray()

    test_vect = np.reshape(testDF.brand_num.values,(-1,1))
    brands_test = enc.transform(test_vect).toarray()
    return (brands_train, brands_val, brands_test)

def build_text_matrix(datadir, dataset_name, tokenizer, df):
    '''
    use bag-of-words representation to convert description_clean into bag-of-words matrix
    '''
    plog("Building text matrices...")
    text_matrix_path=datadir + dataset_name + '_text.mmap'

    bow_matrix = bag_of_words.series_to_bag_of_words(df.description_clean,tokenizer,text_matrix_path,mode="binary")
    return bow_matrix



#Get text data
def build_text_matrices(datadir, tokenizer_path, trainDF, valDF, testDF):
    '''
    use bag-of-words representation to convert descriptions into bag-of-words matrices
    '''
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
    return (bow_train, bow_val, bow_test)

#TODO: modify this to only load a batch of images at a time, from a csv or mmap instead of pickle
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


def get_targets(df):
    '''
    Retrieve target labels and output as float32/int32
    '''
    y1 = df.loc[:,'cat_1_num'].values.astype(np.int32)
    y2 = df.loc[:,'cat_2_num'].values.astype(np.int32)
    y3 = df.loc[:,'cat_3_num'].values.astype(np.int32)
    return y1,y2,y3



def conditional_hstack(brand,bow,image,dataset_name):
    '''
    assumes 'brand' is present.
    if bag of words is not none, hstack it to brand
    if image is not None, hstack it to brand
    '''
    if brand is not None:
        X=brand
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

def merge_data(bows,images,brands):
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
    X_train = conditional_hstack(brands[0],bows[0],images[0],'train')
    X_val = conditional_hstack(brands[1],bows[1],images[1],'val')
    X_test = conditional_hstack(brands[2],bows[2],images[2],'test')

    return X_train.astype(np.float32), X_val.astype(np.float32), X_test.astype(np.float32)

def iterate_batches(trainDF,valDF,testDF, batch_size):
    '''
    iterate through 
    '''

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

    trainDF = shuffle_and_downsample(trainDF,train_samples)
    testDF = shuffle_and_downsample(testDF,test_samples)
    return trainDF,testDF

#TODO: don't return data, save it to a file.
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


    plog("Loading train csv...")
    raw_trainDF = pd.read_csv(trainpath,header = 0, index_col = 0,low_memory = False)
    plog("Loading test csv...")
    raw_testDF = pd.read_csv(testpath,header = 0, index_col = 0,low_memory = False)

    if use_text:
        plog("Loading tokenizer...")
        with open(datadir + 'tokenizer_5000.pkl') as f:
            tokenizer=pkl.load(f)

    full_trainDF = shuffle_and_downsample(raw_trainDF,train_samples)
    full_testDF = shuffle_and_downsample(raw_testDF,test_samples)
    full_trainDF,full_valDF = train_val_split(full_trainDF,val_portion)

    #TODO: cut trainDF, valDF, and testDF into batches
    #TODO: rewrite all of this to operate on train, val, then test, sequentially
    #START HERE
    #then run the following:
    for batch in iterate_batches(full_trainDF, batch_size):
        trainDF = batch

        #Load text data
        if use_text:
            t0 = datetime.now()            
            bow_train = build_text_matrix(tokenizer,trainDF)
            t1 = datetime.now()
            plog("Time to load text: %s" %str(t1-t0))
        else:
            bow_data=None



    #Load text data
    if use_text:
        t0 = datetime.now()
        bow_data=build_text_matrices(datadir, 'tokenizer_5000.pkl', trainDF, valDF, testDF)
        t1 = datetime.now()
        plog("Time to load text: %s" %str(t1-t0))
    else:
        bow_data=None

    #Load image data
    
    if use_images:
        t0 = datetime.now()
        image_data = get_image_matrices(train_imagepath,test_imagepath,trainDF, valDF, testDF)
        t1 = datetime.now()
        plog("Time to load images: %s" %str(t1-t0))
    else:
        image_data=None

    #Load targets
    y1_train,y2_train,y3_train=get_targets(trainDF)
    y1_val,y2_val,y3_val=get_targets(valDF)
    y1_test,y2_test,y3_test=get_targets(testDF)

    #Load brand data
    brand_data = build_brand_matrices(trainDF, valDF, testDF)

    X_train,X_val,X_test = merge_data(bow_data,image_data,brand_data)

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
    iloc0=67500
    iloc1=200000
    save_freq=10
    batch_size = 250
    out_pickle_name=dataset+'_image_features/'+dataset+'_image_features'

    import image_processing
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