'''
download_images_to_directory.py
'''
__author__="Charlie Guthrie"

import os
import sys
from datetime import datetime

import pandas as pd
import numpy as np

#image downloading and processing
from PIL import Image
import urllib, cStringIO

import io
import skimage.transform

def download_and_resize(url):
    '''
    '''
    ext = url.split('.')[-1]
    im = plt.imread(io.BytesIO(urllib.urlopen(url).read()), ext)
    # Resize so smallest dim = 256, preserving aspect ratio
    h, w, _ = im.shape
    if h < w:
        im = skimage.transform.resize(im, (256, w*256/h), preserve_range=True)
    else:
        im = skimage.transform.resize(im, (h*256/w, 256), preserve_range=True)

    # Central crop to 224x224
    h, w, _ = im.shape
    im = im[h//2-112:h//2+112, w//2-112:w//2+112]
    
    #rawim = np.copy(im).astype('uint8')
    rawim = np.astype('uint8')
    return rawim
 
def process_for_rnn(im):
    # Shuffle axes to c01
    im = np.swapaxes(np.swapaxes(im, 1, 2), 0, 1)
    
    # Convert to BGR
    im = im[::-1, :, :]

    im = im - MEAN_IMAGE
    return floatX(im[np.newaxis])

def prep_image(url,idx,dataset,datadir,width,filetype='bmp'):
    '''
    Check to see image file has been downloaded at current size.  If it has not,
    download and resize image. Saves file to datadir/images/[dataset]_[idx]_w[width].[filetype]
    e.g. datadir/images/train_10001_w256.bmp
    
    args:
        url: url of image source
        idx: image row index
        dataset: string 'train' or 'test' or other identifier
        datadir: data directory
        width: desired width of image. Will be resized to width squared
    returns:
        rawim: scaled and cropped image
        im: processed image to input into VGG
    '''
    outpath = datadir + 'images/' + dataset + '_' +  str(idx) + '_w' + str(width) + '.' + filetype
    
    if not os.path.isfile(outpath):
        print "downloading image #%s..." %str(idx)
        try:
            rawim,im = prep_image(url)
        except:
            print "unable to download"
    else:
        print "loading image %i from file..." % idx
        

def get_selected_images(csv_path,first_idx,last_idx,dataset,datadir,width=64,filetype='bmp'):
    '''
    for a given index range, download and resize the images

    args:
        csv_path: path to csv
        first_idx: int or None. last index of range of images to download
        last_idx: int or None. last index of range of images to download
        dataset: string 'train' or 'test' or other identifier

    returns:
        none
    '''
    data = pd.read_csv(csv_path,header = 0, index_col = 0,low_memory = False)
    image_urls = data.large_image_URL.loc[first_idx:last_idx]
    for i,url in image_urls.iteritems():
        download_and_resize(url,i,dataset,datadir,width,filetype)

def main(first_idx,last_idx):
    start_time = datetime.now()

    home = os.path.join(os.path.dirname(__file__),'..')
    #datadir = os.path.join(home,'data') + '/'
    
    #hpc datadir
    datadir = '/scratch/cdg356/spring/data/'
    csv_path = datadir + 'train_set.csv'
    
    get_selected_images(csv_path,first_idx,last_idx,'train',datadir)
    
    end_time = datetime.now()
    runtime = end_time - start_time
    print "Script took",runtime

if __name__ == '__main__':

    if len(sys.argv)<3:
        main(None,None)
    else:
        first_idx = int(sys.argv[1])
        last_idx = int(sys.argv[2])
        main(first_idx,last_idx)
