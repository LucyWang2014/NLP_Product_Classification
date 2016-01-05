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

def image_train_val_split(trainDF, valDF, image_path):
    '''
    split image memmap into training and validation sets
    
    args:
        df: data frame to split
        val_portion: fraction (between 0 and 1) of samples to devote to validation
    returns:
        trainDF
        valDF  
    '''

    #TEMP: shape must be 625001
    shape = (trainDF.shape[0] + valDF.shape[0],4096)
    image_mm = np.memmap(image_path, dtype='float32', mode='r', shape=shape)

    train_image_mm = image_mm[0:trainDF.shape[0]]
    val_image_mm = image_mm[trainDF.shape[0]:trainDF.shape[0] + valDF.shape[0]]

    assert train_image_mm.shape[0] + val_image_mm.shape[0] == image_mm.shape[0]
    assert train_image_mm.shape[1] == val_image_mm.shape[1]
    return train_image_mm, val_image_mm

def get_brand_index(datadir, trainDF,valDF,testDF):
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

    with open(datadir + 'brand_list.pkl','wb') as f:
        pkl.dump(brands,f)

    trainDF.loc[:,'brand_num']=trainDF.brand.apply(apply_brand_index,args=[brands])
    valDF.loc[:,'brand_num']=valDF.brand.apply(apply_brand_index,args=[brands])
    testDF.loc[:,'brand_num']=testDF.brand.apply(apply_brand_index,args=[brands])
    
    trainDF.brand_num.fillna(0, inplace=True)
    valDF.brand_num.fillna(0, inplace=True)
    testDF.brand_num.fillna(0, inplace=True)

def build_brand_matrix(encoder, df):
    '''
    one-hot encode brand indexes
    '''
    plog("Building brand matrix...")
    vect = np.reshape(df.brand_num.values,(-1,1))
    brands_data = encoder.transform(vect).toarray()
    return brands_data

def train_brand_encoder(trainDF):
    enc = OneHotEncoder()
    train_vect = np.reshape(trainDF.brand_num.values,(-1,1))
    enc.fit(train_vect)
    return enc

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

def get_targets(df,mmap_basename='y'):
    '''
    Retrieve target labels as int32 and store into a memmap
    saves:
        memmap of y values
    returns:
        n_values: number of distinct values in each y label
    '''
    y = df.loc[:,['cat_1_num','cat_2_num','cat_3_num']].values.astype(np.int32)

    shape = (y.shape)
    shape_str = "_".join(str(i) for i in shape)
    map_name = datadir + mmap_basename + '_' + shape_str + '.mm'
    mm = np.memmap(map_name, dtype='int32', mode='w+', shape=shape)

    mm[0:shape[0],:] = y[0:shape[0],:]

    keys = ['y_1','y_2','y_3']
    values = np.max(y,axis=0)
    n_values = dict(zip(keys,values))
    return n_values

def merge_data(brand,image,bow):
    '''
    assumes 'brand' is present.
    if bag of words is not none, hstack it to brand
    if image is not None, hstack it to brand
    args:
        brand, bow, image: brand, bag-of-words, and image matrices
    returns:
        X: feature matrix for modeling
    '''
    if brand is not None:
        X=brand
        if bow is not None:
            assert bow.shape[0]==X.shape[0]
            X = np.hstack((X,bow))
        else:
            plog("Bag of words data missing")
        if image is not None:
            assert image.shape[0]==X.shape[0]
            X = np.hstack((X,image))
        else:
            plog("Image data missing from")
    return X.astype(np.float32)

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

def get_features(datadir,df,use_text,use_images,tokenizer,brand_encoder,image_mm,mmap_basename,batch_size):
    '''
    from the initial trainDF, valDF, and testDF dataframes, 
    extract brand, image, and text features in batches and combine into matrices
    Then save matrices to disk using memmaps.

    args:
        datadir: you know by now
        df: trainDF, valDF, or testDF
        use_text: boolean
        use_images: boolean
        tokenizer: pretrained text tokenizer.  trained using bag_of_words.py
        brand_encoder: OneHotEncoder for brand features
        image_path: path to image feature file
        mmap_basename: name of memory map file. Suffix with map's shape will be added later.
        batch_size: size of batches for batch image_processing
    returns:
        none. Saves X feature matrix to disk. 
    '''

    batch_num = 0
    for start_idx in range(0, df.shape[0], batch_size):

        plog("Prepping data for batch %i, starting at index %i" %(batch_num, start_idx))
        #indexes for this batch
        id0=start_idx
        id1=min(start_idx+batch_size,df.shape[0])
        batch = df.iloc[id0:id1,:]

        #Load text data
        if use_text:
            t0 = datetime.now()            
            bow_data = bag_of_words.series_to_bag_of_words(batch.description_clean,tokenizer)
            t1 = datetime.now()
            plog("Time to load text: %s" %str(t1-t0))
        else:
            bow_data=None

        #Load image data
        if use_images:
            t0 = datetime.now()
            image_data = image_mm[id0,id1]
            t1 = datetime.now()
            plog("Time to load images: %s" %str(t1-t0))
        else:
            image_data=None

        #Load brand data
        brand_data = build_brand_matrix(brand_encoder, batch)

        X = merge_data(brand_data,image_data,bow_data)

        #on first batch, initialize memmap
        if id0==0:
            shape = (df.shape[0],X.shape[1])
            shape_str = "_".join(str(i) for i in shape)
            map_name = datadir + mmap_basename + '_' + shape_str + '.mm'
            mm = np.memmap(map_name, dtype='float32', mode='w+', shape=shape)

        #Write to memmap
        mm[id0:id1]=X

        batch_num+=1
    plog("Finished getting features. Shape of final memmap: %s" %str(shape))


def main(datadir,
        train_samples=10000,
        test_samples=1000,
        val_portion=0.1,
        use_images=True,
        use_text=True,
        train_image_fn='train_image_features_625001_4096.mm',
        test_image_fn='test_image_features_100001_4096.mm',
        batch_size=10000,
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
        train_samples = 90
        test_samples = 90
        batch_size=10
    else:
        trainpath = datadir + 'train_set.csv'
        testpath = datadir + 'test_set.csv'


    train_imagepath = datadir + train_image_fn
    test_imagepath = datadir + test_image_fn
    dstart=datetime.now()

    #Load, shuffle, and split the train and test sets
    plog("Loading train csv...")
    raw_trainDF = pd.read_csv(trainpath,header = 0, index_col = 0,low_memory = False)
    plog("Loading test csv...")
    raw_testDF = pd.read_csv(testpath,header = 0, index_col = 0,low_memory = False)
    trainDF_ = shuffle_and_downsample(raw_trainDF,train_samples)
    testDF = shuffle_and_downsample(raw_testDF,test_samples)
    trainDF,valDF = train_val_split(trainDF_,val_portion)

    #Load image memmaps
    train_image_mm, val_image_mm = image_train_val_split(trainDF,valDF,train_imagepath)
    test_image_mm = np.memmap(test_imagepath, dtype='float32', mode='r', shape=(testDF.shape[0],4096))

    #Prepare targets.  Get n_values from training
    n_values = get_targets(trainDF,'y_train')
    get_targets(valDF,'y_val')
    get_targets(testDF,'y_test')

    #Load tokenizer
    if use_text:
        plog("Loading tokenizer...")
        with open('tokenizer_5000.pkl') as f:
            tokenizer=pkl.load(f)
    else:
        tokenizer = None

    #Change brand names to indexes
    get_brand_index(datadir, trainDF ,valDF, testDF)

    #Train brand encoder
    brand_encoder = train_brand_encoder(trainDF)

    #Prepare features
    get_features(datadir,trainDF,use_text,use_images,tokenizer,brand_encoder,train_image_mm,mmap_basename='X_train',batch_size=batch_size)
    get_features(datadir,valDF,use_text,use_images,tokenizer,brand_encoder,val_image_mm,mmap_basename='X_val',batch_size=batch_size)
    get_features(datadir,testDF,use_text,use_images,tokenizer,brand_encoder,test_image_mm,mmap_basename='X_test',batch_size=batch_size)

    dfin = datetime.now()
    plog("Data prep time: %s" %(dfin-dstart))


if __name__ == '__main__':
    home = os.path.join(os.path.dirname(__file__),'..')
    datadir = os.path.join(home,'data') + '/'


    main(datadir,
        train_samples=625001,
        test_samples=100000,
        val_portion=0.1,
        use_images=False,
        use_text=True,
        train_image_fn='head_train_1000_4096.mm',
        test_image_fn='head_test_1000_4096.mm',
        batch_size=10000,
        debug=True)


'''
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
'''