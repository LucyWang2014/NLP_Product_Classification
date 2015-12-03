'''
bag_of_words.py

Converts product descriptions into bag-of-words representations
'''
__author__='Charlie Guthrie'

#TODO: test and debug

import pandas as pd
from keras.preprocessing.text import Tokenizer
import cPickle as pkl

def build_tokenizer(series,nb_words=None,tok_path):
    '''
    
    trains a bag-of-words model from a series
    args:
        series: pandas series made up of strings
        nb_words: None or int. Maximum number of words to work with 
            (if set, tokenization will be restricted to the top nb_words most common words in the dataset)
        tok_path: path for saving the tokenizer
    returns:
        idx: index list for tracking the words
        text_matrix: bag of words matrix representation of text
        dictionary: to identify columns
    '''
    #TODO: check if tokenizer already exists?
    texts = series
    tok = Tokenizer(nb_words=nb_words)
    tok.fit_on_texts(texts)
    with (tok_path, 'wb') as outf:
        pkl.dump(tok,tok_path)
    return tok
    
def series_to_bag_of_words(series,tokenizer,text_matrix_path,mode="binary"):
    '''
    args:
        series: pandas series made up of strings
        tokenizer: keras Tokenizer
        mode:one of "binary", "count", "tfidf", "freq" (default: "binary")
    returns:
        text_matrix:bag of words matrix of shape (len(texts), nb_words)
    '''
    #TODO: check if text matrix path exists?
    texts = series
    idx = series.index
    text_matrix = tokenizer.texts_to_matrix(texts,mode)
    with open(text_matrix_path,'wb') as outf:
        pkl.dump(text_matrix,text_matrix_path)
    return pd.DataFrame(text_matrix, index=series.index)

def main():
    train_df = pd.read_csv('../data/head_train_set.csv',header = 0, index_col = 0,low_memory = False)
    #TODO:
    # build tokenizer from training set
    # build text matrix from training, validation, test sets