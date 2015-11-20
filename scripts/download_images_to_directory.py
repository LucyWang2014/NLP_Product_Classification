'''
download_images.py
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

def download_and_resize(url,idx,dataset,datadir,width=64,filetype='bmp'):
    '''
    Check to see image file has been downloaded at current size.  If it has not,
    download and resize image. Saves file to datadir/images/imx[idx].bmp
    
    args:
        url: url of image source
        idx: image row index
        dataset: string 'train' or 'test' or other identifier
        datadir: data directory
        width: desired width of image. Will be resized to width squared
    returns:
        none
    '''
    outpath = datadir + 'images/' + dataset + str(idx) + '_w' + str(width) + '.' + filetype
    
    if not os.path.isfile(outpath):
        print "getting image #%s..." %str(idx)
        #try:
        img_file = cStringIO.StringIO(urllib.urlopen(url).read())
        img = Image.open(img_file)
        resized = img.thumbnail((width,width), Image.ANTIALIAS)
        img.save(outpath)
       # except:
        #    print "unable to download"
    else:
        print "image # %s already downloaded" %str(idx)

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
