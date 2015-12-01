# coding: utf-8

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pdb
import theano

print "Theano device:",theano.config.device

#dnn requires GPU
import lasagne
from lasagne.layers import InputLayer, DenseLayer, DropoutLayer
from lasagne.layers.dnn import Conv2DDNNLayer as ConvLayer
from lasagne.layers import MaxPool2DLayer as PoolLayer
from lasagne.layers import LocalResponseNormalization2DLayer as NormLayer
from lasagne.utils import floatX


# ### Define the network

# In[ ]:

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


# ### Load the model parameters and metadata
datadir = "/scratch/cdg356/spring/data/"
model = pkl.load(open(datadir+'vgg_cnn_s.pkl'))
CLASSES = model['synset words']
MEAN_IMAGE = model['mean image']

print model['values'][:10]
lasagne.layers.set_all_param_values(output_layer, model['values'])

def get_output(im):
    # Shuffle axes to c01
    im = np.swapaxes(np.swapaxes(im, 1, 2), 0, 1)

    # Convert to BGR
    im = im[::-1, :, :]

    im = im - MEAN_IMAGE
    
    im=floatX(im[np.newaxis])
    
    prob = np.array(lasagne.layers.get_output(output_layer, im, deterministic=True).eval())
    top5 = np.argsort(prob[0])[-1:-6:-1]
    return top5

url="http://content.nordstrom.com/imagegallery/store/product/large/9/_8947109.jpg"
i=0
dataset="train"
width=224
filetype="jpg"

im = prep_image(url,i,dataset,datadir,width,filetype)
top5 = get_output(im)
print top5

pdb.set_trace()

