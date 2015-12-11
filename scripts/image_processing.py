# coding: utf-8

#TODO: 
#batch these up into batches of 256 or 512 images
from utils import create_log,plog
create_log(__file__)

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import theano
import cPickle as pkl
import download_images_to_directory as dl
from datetime import datetime

plog("Theano device: %s" %theano.config.device)

#dnn requires GPU
import lasagne
from lasagne.layers import InputLayer, DenseLayer, DropoutLayer
from lasagne.layers.dnn import Conv2DDNNLayer as ConvLayer
from lasagne.layers import MaxPool2DLayer as PoolLayer
from lasagne.layers import LocalResponseNormalization2DLayer as NormLayer
from lasagne.utils import floatX

# ### Load the model parameters and metadata
def load_pretrained_model(datadir):
    plog("Loading vgg model...")
    model = pkl.load(open(datadir+'vgg_cnn_s.pkl'))
    #CLASSES = model['synset words']
    mean_image = model['mean image']
    return model, mean_image

# ### Define the network
def build_image_network():
    '''
    builds CNN for image feature extraction

    returns:
        network
    '''

    plog("Building lasagne net...")
    net = {}
    net['input'] = InputLayer((None, 3, 224, 224))
    net['conv1'] = ConvLayer(net['input'], num_filters=96, filter_size=7, stride=2)
    net['norm1'] = NormLayer(net['conv1'], alpha=0.0001) # caffe has alpha = alpha * pool_size
    net['pool1'] = PoolLayer(net['norm1'], pool_size=3, stride=3, ignore_border=False)
    net['conv2'] = ConvLayer(net['pool1'], num_filters=256, filter_size=5)
    net['pool2'] = PoolLayer(net['conv2'], pool_size=2, stride=2, ignore_border=False)
    net['conv3'] = ConvLayer(net['pool2'], num_filters=512, filter_size=3, pad=1)
    net['conv4'] = ConvLayer(net['conv3'], num_filters=512, filter_size=3, pad=1)
    net['conv5'] = ConvLayer(net['conv4'], num_filters=512, filter_size=3, pad=1)
    net['pool5'] = PoolLayer(net['conv5'], pool_size=3, stride=3, ignore_border=False)
    net['fc6'] = DenseLayer(net['pool5'], num_units=4096)
    net['drop6'] = DropoutLayer(net['fc6'], p=0.5)
    net['fc7'] = DenseLayer(net['drop6'], num_units=4096)
    net['drop7'] = DropoutLayer(net['fc7'], p=0.5)
    net['fc8'] = DenseLayer(net['drop7'], num_units=1000, nonlinearity=lasagne.nonlinearities.softmax)
    output_layer = net['fc8']
    lasagne.layers.set_all_param_values(output_layer, PRETRAINED_VGG['values'])
    return net

def iterate_minibatches(series, batchsize):
    for start_idx in range(0, series.shape[0] - batchsize + 1, batchsize):
        yield series.iloc[start_idx:start_idx + batchsize]

def prep_for_vgg(url,i,dataset,datadir,width=224,filetype="jpg"):
    '''
    Check to see image file has been downloaded at current size.  If it has not,
    download and resize image. Saves file to datadir/images/[dataset]_[idx]_w[width].[filetype]
    e.g. datadir/images/train_10001_w256.bmp

    args: same as that of dl.prep_image
        url: url of image source
        i: image index
        dataset: string 'train' or 'test' or other identifier
        datadir: data directory
        width: desired width of image. Will be resized to width squared
    returns:
        rawim: scaled and cropped image
    '''
    rawim = dl.prep_image(url,i,dataset,datadir,width,filetype)
    if rawim is None: #If image fails to download, produce 'image' of NaN's with same shape
        im=floatX(np.tile(np.nan,(1,3,width,width)))
    else:
        # Shuffle axes to c01
        im = np.swapaxes(np.swapaxes(rawim, 1, 2), 0, 1)

        # Convert to BGR
        im = im[::-1, :, :]

        im = im - MEAN_IMAGE
        im=floatX(im[np.newaxis])
    return im

def batch_extract_features(batch_series,dataset,datadir,width,filetype):
    '''
    take batch_series and return dataframe of image features with shape (batchDF.shape[0],4096)
    args:
        batch_series:
        dataset: "train" or "test"
        datadir
        width:224
        filetype:jpg
    returns:
        featureDF: keeps original indexes, but has different column for each image feature
    '''
    image_urls = batch_series
    indexes = batch_series.index
    first=True
    for i,url in image_urls.iteritems():
        im = prep_for_vgg(url,i,dataset,datadir,width,filetype)
        if first==True:
            images = im
            first=False
        else:
            images = np.vstack((images,im))
    
    #image_features = np.reshape(images,(images.shape[0],-1))
    #get last layer from vgg model.  This part takes ~1-4 seconds
    image_features = np.array(lasagne.layers.get_output(IMAGE_NET['fc7'], images, deterministic=True).eval())

    featureDF = pd.DataFrame(image_features, index=[indexes]) 
    return featureDF

def get_selected_image_features(df,
                                datadir,
                                dataset,
                                iloc0,
                                iloc1,
                                save_freq,
                                out_pickle_name='image_features.pkl',
                                batch_size=256,
                                width=224,
                                filetype='jpg'):
    '''
    for a given index range, download and resize the images,
    then save to directory

    args:
        df: dataframe where image urls are
        iloc0: int or None. first iloc of range of images to download
        iloc1: int or None. last iloc of range of images to download
        save_freq: how many batches before saving
        out_pickle_name: name of outfile
        batch_size: rows per batch
        dataset: string 'train' or 'test' or other identifier

    returns:
        none
    '''
    plog("Beginning feature extraction...")
    assert iloc0<=df.shape[0]
    assert iloc1<=df.shape[0]
    image_urls = df.large_image_URL.iloc[iloc0:iloc1]
    iloc=iloc0
    prev_iloc = iloc0
    batch_num=0
    featureDF = pd.DataFrame()
    
    for batch in iterate_minibatches(image_urls,batch_size):
        plog("extracting image features for batch %i, iloc %i" %(batch_num,iloc))
        batch_featureDF = batch_extract_features(batch,dataset,datadir,width,filetype)
        featureDF=featureDF.append(batch_featureDF,verify_integrity=True)
        
        iloc+=batch_size
        batch_num+=1
        
        if iloc>iloc0 and (batch_num%save_freq==0 or iloc>=iloc1-1):
            plog("Saving from image iloc %i to image iloc %i" %(prev_iloc,iloc))
            with open(datadir+out_pickle_name + '_' + str(prev_iloc) + '_' + str(iloc)+'.pkl','wb') as outf:
                pkl.dump(featureDF,outf)  
            prev_iloc = iloc

            #reset featureDF to save memory
            featureDF = pd.DataFrame()



DATADIR = "../data/"
PRETRAINED_VGG, MEAN_IMAGE = load_pretrained_model(DATADIR)
IMAGE_NET = build_image_network()

if __name__ == '__main__':
    from datetime import datetime
    start_time = datetime.now()

    csv_name='train_set.csv'
    iloc0=None
    iloc1=None
    dataset="train"
    datadir = DATADIR
    out_pickle_name=dataset+'_image_features'

    end_time = datetime.now()
    runtime = end_time - start_time
    plog("script runtime: "+str(runtime))
