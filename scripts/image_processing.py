# coding: utf-8

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pdb
import theano
import cPickle as pkl
import download_images_to_directory as dl
import logging as log
log.basicConfig(filename='../logs/image_processing.log',level=log.DEBUG)

print "Theano device:",theano.config.device
log.info("Theano device: %s" %theano.config.device)

#dnn requires GPU
import lasagne
from lasagne.layers import InputLayer, DenseLayer, DropoutLayer
from lasagne.layers.dnn import Conv2DDNNLayer as ConvLayer
from lasagne.layers import MaxPool2DLayer as PoolLayer
from lasagne.layers import LocalResponseNormalization2DLayer as NormLayer
from lasagne.utils import floatX

# ### Load the model parameters and metadata
def load_pretrained_model(datadir):
    print "Loading vgg model..."
    log.info ("Loading vgg model...")
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

    print "Building lasagne net..."
    log.info("Building lasagne net...")
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

def extract_image_features(url,i,dataset,datadir,width=224,filetype="jpg"):
    '''
    Check to see image file has been downloaded at current size.  If it has not,
    download and resize image. Saves file to datadir/images/[dataset]_[idx]_w[width].[filetype]
    e.g. datadir/images/train_10001_w256.bmp

    args: same as that of dl.prep_image
        url: url of image source
        idx: image row index
        dataset: string 'train' or 'test' or other identifier
        datadir: data directory
        width: desired width of image. Will be resized to width squared
    returns:
        rawim: scaled and cropped image
    '''
    if i%10000==0:
        print "Extracting features from image index",i
        log.info("Extracting features from image index %i" %i)
    rawim = dl.prep_image(url,i,dataset,datadir,width,filetype)

    # Shuffle axes to c01
    im = np.swapaxes(np.swapaxes(rawim, 1, 2), 0, 1)

    # Convert to BGR
    im = im[::-1, :, :]

    im = im - MEAN_IMAGE
    im=floatX(im[np.newaxis])
    
    #get last layer from vgg model
    ll = np.array(lasagne.layers.get_output(IMAGE_NET['fc7'], im, deterministic=True).eval())
    return ll

def get_selected_image_features(df,
                                first_idx,
                                last_idx,
                                dataset,
                                datadir,
                                out_pickle_name='image_features.pkl',
                                width=224,
                                filetype='jpg'):
    '''
    for a given index range, download and resize the images,
    then save to directory

    args:
        df: dataframe where image urls are
        first_idx: int or None. last index of range of images to download
        last_idx: int or None. last index of range of images to download
        dataset: string 'train' or 'test' or other identifier

    returns:
        none
    '''
    data = df
    image_urls = data.large_image_URL.loc[first_idx:last_idx]
    featureDF = pd.DataFrame()
    ticker=1
    #iterate through index and url
    for i,url in image_urls.iteritems():
        image_feature = extract_image_features(url,i,dataset,datadir,width,filetype)
        featureDF.loc[i,'image_feature']=image_feature.astype(object) #NOTE: may need to convert back to float32 with x=featureDF.loc[i,'image_feature'].astype(np.float32)
        if ticker%100000==0:
            print "Saving up to image index",i
            log.info("Saving up to image index %i" %i)
            with open(datadir+out_pickle_name + str(i)+'.pkl','wb') as outf:
                pkl.dump(featureDF,outf)  
            ticker+=1

    with open(datadir+out_pickle_name+'.pkl','wb') as outf:
        pkl.dump(featureDF,outf)

    return featureDF

    #with open('image_feature_test.csv','wb') as outf:
    #    featureDF.to_csv(outf, header=True, index=True)

DATADIR = "/scratch/cdg356/spring/data/"
PRETRAINED_VGG, MEAN_IMAGE = load_pretrained_model(DATADIR)
IMAGE_NET = build_image_network()

if __name__ == '__main__':
    from datetime import datetime
    start_time = datetime.now()

    csv_name='train_set.csv'
    first_idx=None
    last_idx=None
    dataset="train"
    datadir = DATADIR
    out_pickle_name='train_image_features'

    get_selected_image_features(csv_name,
                                first_idx,
                                last_idx,
                                dataset,
                                datadir,
                                out_pickle_name,
                                width=224,
                                filetype='jpg')
    end_time = datetime.now()
    runtime = end_time - start_time
    print "script runtime: ",runtime
    log.info("script runtime: "+str(runtime))
