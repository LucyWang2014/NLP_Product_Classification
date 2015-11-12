import cPickle
import gzip
import os

import numpy
import theano
import pdb

def prepare_data(seqs, labels, maxlen=None):
    """Create the matrices from the datasets.

    This pad each sequence to the same lenght: the lenght of the
    longuest sequence or maxlen.

    if maxlen is set, we will cut all sequence to this maximum
    lenght.

    This swap the axis!
    """
    # x: a list of sentences
    lengths = [len(s) for s in seqs]
    #pdb.set_trace()
    if maxlen is not None:
        new_seqs = []
        new_labels = []
        new_lengths = []
        for l, s, y in zip(lengths, seqs, labels):
            if l < maxlen:
                new_seqs.append(s)
                new_labels.append(y)
                new_lengths.append(l)
        lengths = new_lengths
        labels = new_labels
        seqs = new_seqs

        if len(lengths) < 1:
            return None, None, None

    n_samples = len(seqs)
    maxlen = numpy.max(lengths)

    x = numpy.zeros((maxlen, n_samples)).astype('int64')
    x_mask = numpy.zeros((maxlen, n_samples)).astype(theano.config.floatX)
    for idx, s in enumerate(seqs):
        x[:lengths[idx], idx] = s
        x_mask[:lengths[idx], idx] = 1.

    return x, x_mask, labels


def get_dataset_file(dataset, default_dataset):
    '''Look for it as if it was a full path, if not, try local file,
    if not try in the data directory.

    Download dataset if it is not present

    '''
    data_dir, data_file = os.path.split(dataset)
    if data_dir == "" and not os.path.isfile(dataset):
        # Check if dataset is in the data directory.
        new_path = os.path.join(
            os.path.split(__file__)[0],
            "..",
            "data",
            dataset
        )
        if os.path.isfile(new_path) or data_file == default_dataset:
            dataset = new_path

    if (not os.path.isfile(dataset)) and data_file == default_dataset:
        raise Exception('no file available')
    return dataset


def load_data(path="nordstrom", n_words=100000, valid_portion=0.1, maxlen=None,
              sort_by_len=True):
    '''Loads the dataset

    :type path: String
    :param path: The path to the dataset (here Nordstrom)
    :type n_words: int
    :param n_words: The number of word to keep in the vocabulary.
        All extra words are set to unknow (1).
    :type valid_portion: float
    :param valid_portion: The proportion of the full train set used for
        the validation set.
    :type maxlen: None or positive int
    :param maxlen: the max sequence length we use in the train/valid set.
    :type sort_by_len: bool
    :name sort_by_len: Sort by the sequence lenght for the train,
        valid and test set. This allow faster execution as it cause
        less padding per minibatch. Another mechanism must be used to
        shuffle the train set at each epoch.

    '''

    #############
    # LOAD DATA #
    #############

    # Load the dataset
    #path = get_dataset_file(
    #   path, "nordstrom")


    #if path.endswith(".gz"):
    #    f = gzip.open(path, 'rb')
    #else:
    f_train = open(path + '_train.pkl', 'rb')
    f_test = open(path + '_test.pkl', 'rb')
    dictionary = open(path + '.dict.pkl','rb')

    train_set = cPickle.load(f_train)
    test_set = cPickle.load(f_test)
    dictionary = cPickle.load(dictionary)

    f_train.close()
    f_test.close()
    
    if maxlen:
        new_train_set_x_1 = []
        new_train_set_x_2 = []
        new_train_set_x_3 = []
        new_train_set_y_1 = []
        new_train_set_y_2 = []
        new_train_set_y_3 = []
        for x_1, x_2, x_3, y_1,y_2,y_3 in zip(train_set[0], train_set[1],train_set[2],train_set[3],train_set[4],train_set[5]):
            if len(x_1) < maxlen:
                new_train_set_x_1.append(x_1)
                new_train_set_x_2.append(x_2)
                new_train_set_x_3.append(x_3)
                new_train_set_y_1.append(y_1)
                new_train_set_y_2.append(y_2)
                new_train_set_y_3.append(y_3)
        train_set = (new_train_set_x_1, new_train_set_x_2, new_train_set_x_3,new_train_set_y_1,new_train_set_y_2,new_train_set_y_3)
        del new_train_set_x_1, new_train_set_x_2, new_train_set_x_3, new_train_set_y_1,new_train_set_y_2, new_train_set_y_3
    

    # split training set into validation set
    train_set_x_1, train_set_x_2, train_set_x_3, train_set_y_1, train_set_y_2, train_set_y_3 = train_set
    n_samples = len(train_set_x_1)
    sidx = numpy.random.permutation(n_samples)
    n_train = int(numpy.round(n_samples * (1. - valid_portion)))
    valid_set_x_1 = [train_set_x_1[s] for s in sidx[n_train:]]
    valid_set_x_2 = [train_set_x_2[s] for s in sidx[n_train:]]
    valid_set_x_3 = [train_set_x_3[s] for s in sidx[n_train:]]
    valid_set_y_1 = [train_set_y_1[s] for s in sidx[n_train:]]
    valid_set_y_2 = [train_set_y_2[s] for s in sidx[n_train:]]
    valid_set_y_3 = [train_set_y_3[s] for s in sidx[n_train:]]
    train_set_x_1 = [train_set_x_1[s] for s in sidx[:n_train]]
    train_set_x_2 = [train_set_x_2[s] for s in sidx[:n_train]]
    train_set_x_3 = [train_set_x_3[s] for s in sidx[:n_train]]
    train_set_y_1 = [train_set_y_1[s] for s in sidx[:n_train]]
    train_set_y_2 = [train_set_y_2[s] for s in sidx[:n_train]]
    train_set_y_3 = [train_set_y_3[s] for s in sidx[:n_train]]

    train_set = (train_set_x_1, train_set_x_2, train_set_x_3, train_set_y_1,train_set_y_2,train_set_y_3)
    valid_set = (valid_set_x_1, valid_set_x_2, valid_set_x_3, valid_set_y_1,valid_set_y_2,valid_set_y_3)

    def remove_unk(x):
        return [[1 if w >= n_words else w for w in sen] for sen in x]

    test_set_x_1, test_set_x_2, test_set_x_3, test_set_y_1,test_set_y_2,test_set_y_3 = test_set
    valid_set_x_1, valid_set_x_2, valid_set_x_3, valid_set_y_1, valid_set_y_2,valid_set_y_3 = valid_set
    train_set_x_1, train_set_x_2, train_set_x_3, train_set_y_1, train_set_y_2,train_set_y_3 = train_set

    train_set_x_1 = remove_unk(train_set_x_1)
    valid_set_x_1 = remove_unk(valid_set_x_1)
    test_set_x_1 = remove_unk(test_set_x_1)

    def len_argsort(seq):
        return sorted(range(len(seq)), key=lambda x: len(seq[x]))

    if sort_by_len:
        sorted_index = len_argsort(test_set_x_1)
        test_set_x_1 = [test_set_x_1[i] for i in sorted_index]
        test_set_x_2 = [test_set_x_2[i] for i in sorted_index]
        test_set_x_3 = [test_set_x_3[i] for i in sorted_index]
        test_set_y_1 = [test_set_y_1[i] for i in sorted_index]
        test_set_y_2 = [test_set_y_2[i] for i in sorted_index]
        test_set_y_3 = [test_set_y_3[i] for i in sorted_index]

        sorted_index = len_argsort(valid_set_x_1)
        valid_set_x_1 = [valid_set_x_1[i] for i in sorted_index]
        valid_set_x_2 = [valid_set_x_2[i] for i in sorted_index]
        valid_set_x_3 = [valid_set_x_3[i] for i in sorted_index]
        valid_set_y_1 = [valid_set_y_1[i] for i in sorted_index]
        valid_set_y_2 = [valid_set_y_2[i] for i in sorted_index]
        valid_set_y_3 = [valid_set_y_3[i] for i in sorted_index]

        sorted_index = len_argsort(train_set_x_1)
        train_set_x_1 = [train_set_x_1[i] for i in sorted_index]
        train_set_x_2 = [train_set_x_2[i] for i in sorted_index]
        train_set_x_3 = [train_set_x_3[i] for i in sorted_index]
        train_set_y_1 = [train_set_y_1[i] for i in sorted_index]
        train_set_y_2 = [train_set_y_2[i] for i in sorted_index]
        train_set_y_3 = [train_set_y_3[i] for i in sorted_index]

    train = (train_set_x_1, train_set_x_2, train_set_x_3, train_set_y_1,train_set_y_2,train_set_y_3)
    valid = (valid_set_x_1, valid_set_x_2, valid_set_x_3, valid_set_y_1, valid_set_y_2,valid_set_y_3)
    test = (test_set_x_1, test_set_x_2, test_set_x_3, test_set_y_1,test_set_y_2,test_set_y_3)

    return train, valid, test, dictionary
