"""
This script is what created the dataset pickled.

1) You need to download this file and put it in the same directory as this file.
https://github.com/moses-smt/mosesdecoder/raw/master/scripts/tokenizer/tokenizer.perl . Give it execution permission.

2) Get the dataset from  and extract it in the current directory.

3) Then run this script.
"""

import numpy
#import pdb
import cPickle as pkl

from collections import OrderedDict

import glob
import os

from subprocess import Popen, PIPE

import pandas as pd

# tokenizer.perl is from Moses: https://github.com/moses-smt/mosesdecoder/tree/master/scripts/tokenizer
tokenizer_cmd = ['./mosesdecoder/scripts/tokenizer/tokenizer.perl', '-l', 'en', '-q', '-']


def tokenize(sentences):
    """
    Tokenizes sentences by removing irrelevant punctuation, etc.
    Uses the mosesdecoder in some way.

    Args:
        sentences: list of sentences
    """

    print 'Tokenizing..',
    text = "\n".join(sentences)
    tokenizer = Popen(tokenizer_cmd, stdin=PIPE, stdout=PIPE)
    tok_text, _ = tokenizer.communicate(text)
    toks = tok_text.split('\n')[:-1]
    print 'Done'

    return toks


def build_dict(path):
    """
    Build dictionary

    Args:
        path: (string) path of training csv

    Returns:
        worddict: dictionary
        cat_1_dict
        cat_2_dict
    """
    train_df = pd.read_csv(path,header = 0, index_col = 0,low_memory = False)

    cat_1 = list(train_df.cat_1.astype(str))
    cat_2 = list(train_df.cat_2.astype(str))

    descriptions = list(train_df.description_clean.astype(str) + 
                    train_df.brand.astype(str))


    descriptions = tokenize(descriptions)
    #cat_1 = tokenize(cat_1)
    #cat_2 = tokenize(cat_2)

    #pdb.set_trace()

    print 'Building dictionary..',
    wordcount = dict()
    for ss in descriptions:
        words = ss.strip().lower().split()
        
        for w in words:
            if w not in wordcount:
                wordcount[w] = 1
            else:
                wordcount[w] += 1

    cat_1_count = dict()
    for cat in cat_1:
        if cat not in cat_1_count:
            cat_1_count[cat] = 1
        else:
            cat_1_count[cat] += 1

    cat_2_count = dict()
    for cat in cat_2:
        if cat not in cat_1_count:
            cat_2_count[cat] = 1
        else:
            cat_2_count[cat] += 1

    totalcount = wordcount.copy()
    totalcount.update(cat_1_count)
    totalcount.update(cat_2_count)

    counts = totalcount.values()
    keys = totalcount.keys()

    desc_counts = wordcount.values()
    sorted_idx = numpy.argsort(counts)[::-1]


    worddict = dict()

    cat1_len = len(cat_1_count)
    cat2_len = len(cat_2_count)

    cat_1_counter = 0
    cat_2_counter = 0

    cat_1_dict = dict()
    cat_2_dict = dict()
    idx = 0
    for ss in sorted_idx:
 
        if keys[ss] in cat_1_count:
            worddict[keys[ss]] = cat_1_counter + 2 # leave 0 and 1 (UNK)
            cat_1_dict[train_df.cat_1_num[train_df.cat_1 == keys[ss]].iloc[0]] = keys[ss] 
            cat_1_counter += 1
        elif keys[ss] in cat_2_count:
            worddict[keys[ss]] = cat_2_counter + 2 + cat1_len #leave 0 and 1, and cat_1's
            cat_2_dict[train_df.cat_2_num[train_df.cat_2 == keys[ss]].iloc[0]] = keys[ss] 
            cat_2_counter += 1
        else:
            worddict[keys[ss]] = idx + 2 + cat1_len + cat2_len #leave 0 and 1, and cat_1's + cat_2's
            idx += 1
        
    print numpy.sum(counts), ' total words ', len(keys), ' unique words'

    return worddict, cat_1_dict, cat_2_dict


def grab_data(path, dictionary):

    data = pd.read_csv(path,index_col = 0, header = 0,low_memory=False)

    sentences = list(data.description_clean.astype(str) + 
        data.brand.astype(str))

    sentences = tokenize(sentences)

    seqs = [None] * len(sentences) 

    for idx, ss in enumerate(sentences):
        words = ss.strip().lower().split()
        seqs[idx] = [dictionary[w] if w in dictionary else 1 for w in words]        

    cat_1 = []
    cat_2 = []
    for l_1,l_2 in zip(data.cat_1,data.cat_2):
        cat_1.append(dictionary[str(l_1)])
        cat_2.append(dictionary[str(l_2)])

    cat_1 = zip(data.cat_1_num,cat_1)
    cat_2 = zip(data.cat_2_num,cat_2)

    label_1 = list(data.cat_1_num)
    label_2 = list(data.cat_2_num)
    label_3 = list(data.cat_3_num)

    return seqs, cat_1, cat_2, label_1, label_2, label_3

def main():
    # Get the dataset from http://ai.stanford.edu/~amaas/data/sentiment/
    #folder_path = '/Users/LittleLucy/Google Drive/MSDS/2015Fall/DSGA3001_NLP_Distributed_Representation/Project/'

    '''
    Let's use relative paths instead, so we don't have to switch back and forth depending on the machine.
    '''
    dataset_path='data/'

    dictionary,cat_1_dict, cat_2_dict = build_dict(dataset_path + 'train_set.csv')

    train_x, train_cat_1, train_cat_2, train_y_1, train_y_2, train_y_3 = grab_data(dataset_path+'train_set.csv', dictionary)

    test_x, test_cat_1, test_cat_2, test_y_1, test_y_2, test_y_3 = grab_data(dataset_path+'test_set.csv', dictionary)

    f = open(dataset_path + 'encode_cats/nordstrom_train.pkl', 'wb')
    pkl.dump((train_x, train_cat_1, train_cat_2, train_y_1, train_y_2, train_y_3), f, -1)
    f.close()

    f = open(dataset_path + 'encode_cats/nordstrom_test.pkl','wb')
    pkl.dump((test_x, test_cat_1, test_cat_2, test_y_1, test_y_2, test_y_3), f, -1)
    f.close()

    f = open(dataset_path + 'encode_cats/nordstrom.dict.pkl', 'wb')
    pkl.dump((dictionary,cat_1_dict, cat_2_dict), f, -1)
    f.close()

if __name__ == '__main__':
    main()
