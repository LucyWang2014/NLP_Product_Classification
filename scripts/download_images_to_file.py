'''
download_images.py
'''
__author__="Charlie Guthrie"

import os,sys
import cPickle as pkl
from datetime import datetime

import pandas as pd
import numpy as np

#image downloading and processing
from PIL import Image
import urllib, cStringIO

def download_and_resize_image(url,width):
    '''
    download and resize, convert to numpy array
    
    args:
        url: url of image
        width: desired width of image
    returns:
        np.array formatted version of image
    '''
    print "getting image from url: %s ..." %url
    try:
        img_file = cStringIO.StringIO(urllib.urlopen(url).read())
        img = Image.open(img_file)
        resized = img.thumbnail((width,width), Image.ANTIALIAS)
        return np.array(img)
    except:
        print "image not downloaded"
        print sys.exc_info()[0]
        return None

def add_all_images(datadir,infile_name,outfile_name,width=64):
    '''
    download and resize images and insert them into the data frame
    also saves new data frame to disc

    args:
        datadir: data directory
        infile_name: csv input file name
        outfile_name: csv output file name
        width: desired width of image
    returns:
        data frame with image
    '''
    csv_path = datadir + infile_name
    textDF = pd.read_csv(csv_path,header = 0, index_col = 0,low_memory = False)
    imageDF = pd.DataFrame(textDF.large_image_URL.apply(download_and_resize_image,args=(width,)))
    with open(datadir + outfile_name, 'wb') as outf:
        pkl.dump(imageDF,outf)
    #imageDF.to_csv(datadir + outfile_name + '.csv')
    return textDF,imageDF

def main(infile_name, outfile_name):
    start_time = datetime.now()

    HOME = os.path.join(os.getcwd(),'..')
    
    #Local machine
    #DATADIR = os.path.join(HOME,'data') + '/'
    #HPC Datadir
    DATADIR = '/scratch/cdg356/spring/data/'

    #get all the images
    add_all_images(DATADIR,infile_name,outfile_name)
    
    end_time = datetime.now()
    runtime = end_time - start_time
    print "Script took",runtime

if __name__ == '__main__':
    if len(sys.argv)<3:
        print 'usage: python download_images_to_file.py "infile_name.csv" "outfile_name.pkl"'
    else:
        infile_name = sys.argv[1]
        outfile_name = sys.argv[2]
        main(infile_name,outfile_name)
