import numpy

def one_hot_encode_features(data, n_values = None):

    if n_values is None:
        n_values = max(data) + 1
    n_samples = len(data)

    encoded_features = numpy.zeros((n_samples,n_values)).astype('int32')

    for idx, seq in enumerate(data):
        if type(seq) == list:
            for w in seq: 
                if w < n_values:
        	       encoded_features[idx][w] = 1
        else:
           encoded_features[idx][seq] = 1 

    return encoded_features