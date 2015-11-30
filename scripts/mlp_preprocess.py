import cPickle as pkl
import os
import sys
import time

from subprocess import Popen, PIPE

#from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import numpy
#from scipy.sparse import hstack, lil_matrix

import pdb

# tokenizer.perl is from Moses: https://github.com/moses-smt/mosesdecoder/tree/master/scripts/tokenizer
tokenizer_cmd = ['./mosesdecoder/scripts/tokenizer/tokenizer.perl', '-l', 'en', '-q', '-']


def tokenize(sentences):
    """
    Tokenizes sentences by removing irrelevant punctuation, etc.
    Uses the mosesdecoder in some way.

    Args:
        sentences: list of sentences

    Returns:
        toks: list of tokens?
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
    Get word counts from the descriptions
    Get counts by category level 1
    Get counts by category level 2

    Args:
        path: (string) path of training csv

    Returns:
        worddict: dict of word and word count
        cat_1_dict: dict of category with counts for each
        cat_2_dict: dict of category with counts for each
    """
    train_df = pd.read_csv(path,header = 0, index_col = 0,low_memory = False)

    descriptions = list(train_df.description_clean.astype(str))

    print 'Tokenizing descriptions...'
    descriptions = tokenize(descriptions)
    print "Done"

    print 'Getting descriptions word count..',
    wordcount = dict()
    for ss in descriptions:
        words = ss.strip().lower().split()
        
        for w in words:
            if w not in wordcount:
                wordcount[w] = 1
            else:
                wordcount[w] += 1
    print 'Done'

    counts = wordcount.values()
    keys = wordcount.keys()
    sorted_idx = numpy.argsort(counts)[::-1]

    worddict = dict()

    idx = 0
    for ss in sorted_idx:
    	worddict[keys[ss]] = idx + 2 #leave 0 and 1, and cat_1's + cat_2's
    	idx += 1
        
    print numpy.sum(counts), ' total words ', len(keys), ' unique words'

    return worddict


def grab_bag_of_words(path, dictionary):
	
    print 'loading data...'
    data = pd.read_csv(path,index_col = 0, header = 0,low_memory=False)

    sentences = list(data.description_clean.astype(str))
    brands = list(data.brand_num.astype(int))
    label_1 = list(data.cat_1_num)
    label_2 = list(data.cat_2_num)
    label_3 = list(data.cat_3_num)

    seqs = [None] * len(sentences)
    sentences = tokenize(sentences)
    for idx, ss in enumerate(sentences):
    	words = ss.strip().lower().split()
        seqs[idx] = [dictionary[w] if w in dictionary else 1 for w in words]

    '''
    # Initialize the "CountVectorizer" object, which is scikit-learn's
    # bag of words tool.
    print 'building vectorizer...'
    vectorizer = CountVectorizer(analyzer = "word", 
    								tokenizer = None, 
    								preprocessor = None,
                                 	stop_words = None, 
                                 	max_features = 5000) 

    # fit_transform() does two functions: First, it fits the model
    # and learns the vocabulary; second, it transforms our training data
    # into feature vectors. The input to fit_transform should be a list of 
    # strings.
    print 'encoding features...'
    data_features = vectorizer.fit_transform(sentences)

    # Numpy arrays are easy to work with, so convert the result to an 
    # array
    data_features = lil_matrix(data_features)

    brands_features = lil_matrix(one_hot_encode_features(brands))
    label_1 = one_hot_encode_features(label_1)
    label_2 = one_hot_encode_features(label_2)
    label_3 = one_hot_encode_features(label_3)

    print 'concatenating shared data features...'
    data_features = hstack((data_features,brands_features))

    '''

    return seqs, brands, label_1, label_2, label_3

def main(data_directory):
    '''
    Let's use relative paths instead, so we don't have to switch back and forth depending on the machine.
    '''
    start_time = time.time()
    home = os.path.join(os.path.dirname(__file__),'..')
    print "file",__file__
    dataset_path = os.path.join(home,'data') + '/'
    print "home",home
    print "dataset_path",dataset_path

    print "building dictionary..."
    dictionary = build_dict(dataset_path + 'train_set.csv')
    
    print 'building training set...'
    train_desc, train_brands, train_label_1, train_label_2, train_label_3 = grab_bag_of_words(dataset_path+'train_set.csv', dictionary)

    print 'building test set...'
    test_desc, test_brands, test_label_1, test_label_2, test_label_3 = grab_bag_of_words(dataset_path+'test_set.csv',dictionary)

    #Create directory if not present
    if not os.path.exists(dataset_path + data_directory):
        os.makedirs(dataset_path + data_directory)
    
    print 'saving pickle files...'
    f = open(dataset_path + data_directory + '/nordstrom_train.pkl', 'wb')
    pkl.dump((train_desc, train_brands, train_label_1, train_label_2, train_label_3), f, -1)
    f.close()

    f = open(dataset_path + data_directory + '/nordstrom_test.pkl','wb')
    pkl.dump((test_desc, test_brands, test_label_1, test_label_2, test_label_3), f, -1)
    f.close()

    end_time = time.time()

    print 'The code run for %f min %f sec' % ((end_time - start_time) / 60, 
    	(end_time - start_time) % 60)

if __name__ == '__main__':
	print 'saving to ' + sys.argv[1]
	main(sys.argv[1])






