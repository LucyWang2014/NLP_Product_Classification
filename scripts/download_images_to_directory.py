'''
download_images_to_directory.py
'''
__author__="Charlie Guthrie"

import os,sys
from datetime import datetime
import pandas as pd
import numpy as np
import cPickle as pkl

#image downloading and processing
import urllib
import io
import matplotlib.pyplot as plt
import skimage.transform

def download_and_resize(url,width):
    '''
    download image url, resize and crop to [width]
    args:
        url: url of image
        width: width of square image
    returns:
        rawim: image cropped and resized to width square
    '''
    ext = url.split('.')[-1]
    im = plt.imread(io.BytesIO(urllib.urlopen(url).read()), ext)
    # Resize so smallest dim = 256, preserving aspect ratio
    h, w, _ = im.shape
    if h < w:
        im = skimage.transform.resize(im, (width, w*width/h), preserve_range=True)
    else:
        im = skimage.transform.resize(im, (h*width/w, width), preserve_range=True)

    # Central crop to width x width
    halfwidth = width/2
    h, w, _ = im.shape
    im = im[h//2-halfwidth:h//2+halfwidth, w//2-halfwidth:w//2+halfwidth]

    #rawim = np.copy(im).astype('uint8')
    rawim = im.astype('uint8')
    return rawim

def prep_image(url,idx,dataset,datadir,width=224,filetype='jpg'):
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
    '''
    outpath = datadir + 'images/' + dataset + '_' +  str(idx) + '_w' + str(width) + '.' + filetype

    if not os.path.isfile(outpath):
        print "downloading image #%s..." %str(idx)
        try:
            rawim = download_and_resize(url,width)
            plt.imsave(outpath,rawim)
            return rawim
        except:
            print "unable to download"
    else:
        print "Image %i already downloaded. Loading from file..." % idx
        rawim = plt.imread(outpath)
        return rawim
        
def get_selected_images(csv_name,first_idx,last_idx,dataset,datadir,width=224,filetype='jpg'):
    '''
    for a given index range, download and resize the images,
    then save to directory

    args:
        csv_name: name of csv (assumed to be in datadir)
        first_idx: int or None. last index of range of images to download
        last_idx: int or None. last index of range of images to download
        dataset: string 'train' or 'test' or other identifier

    returns:
        none
    '''
    data = pd.read_csv(datadir+csv_name,header = 0, index_col = 0,low_memory = False)
    image_urls = data.large_image_URL.loc[first_idx:last_idx]
    for i,url in image_urls.iteritems():
        prep_image(url,i,dataset,datadir,width,filetype)

def main(csv_name,dataset,first_idx,last_idx):
    start_time = datetime.now()

    home = os.path.join(os.path.dirname(__file__),'..')
    #local datadir
    #datadir = os.path.join(home,'data') + '/'

    #hpc datadir
    datadir = '/scratch/cdg356/spring/data/'
    
    get_selected_images(csv_name,first_idx,last_idx,dataset,datadir)

    end_time = datetime.now()
    runtime = end_time - start_time
    print "Script took",runtime

if __name__ == '__main__':

    if len(sys.argv)<2:
        print "usage: python csv_file dataset [first_idx] [last_idx]"
        print "example: python train_set.csv train"
        main('train_set.csv','train',None,None)
    elif len(sys.argv)<4:
        csv_name = sys.argv[1]
        dataset = sys.argv[2]
        main(csv_name,dataset,None,None)
    else:
        csv_name = sys.argv[1]
        dataset = sys.argv[2]
        first_idx = int(sys.argv[3])
        last_idx = int(sys.argv[4])
        main(csv_name,dataset,first_idx,last_idx)
