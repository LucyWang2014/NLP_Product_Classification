"""
Adapted from the MNIST example codes in Lasagne

"""

from __future__ import print_function

import sys
import os
import time
import cPickle
import collections

import numpy as np
import theano
import theano.tensor as T

import lasagne

from mlp_functions import one_hot_encode_features

import pdb


# ################## load the data ##################

def get_data(
    path = '../data/mlp_data/', #This is the path for the dictionaries to use
    test_size=100,  # If >0, we keep only this number of test example.
    train_size = 2500, # If >0, we keep only this number of train example.
    valid_portion = 0.1, # percent of training data for validation set
    desc_n_values = 5000, # percent of training data for validation set
    ):
    '''
    returns text data and n_values
    '''

    f_train = open(path + 'nordstrom_train.pkl', 'rb')
    f_test = open(path + 'nordstrom_test.pkl', 'rb')
    
    train_set = cPickle.load(f_train)
    test = cPickle.load(f_test)

    f_train.close()
    f_test.close()

    #set the number of uniques for each variable
    keys = ['desc','brands','y_1','y_2','y_3']
    values = [desc_n_values,max(train_set[1])+1,max(train_set[2])+1,max(train_set[3])+1, max(train_set[4])+1]
    n_values = collections.OrderedDict(zip(keys,values))

    # split training set into validation set
    train_desc, train_brands, train_y_1, train_y_2, train_y_3 = train_set
    n_samples = len(train_y_1)
    #PROBLEM? due to missing numbers in training index, max(training index) may be greater than len(train_y_1)
    sidx = np.random.permutation(n_samples)
    n_train = int(np.round(n_samples * (1. - valid_portion)))
    valid_desc = [train_desc[s] for s in sidx[n_train:]]
    valid_brands = [train_brands[s] for s in sidx[n_train:]]
    valid_y_1 = np.array([train_y_1[s] for s in sidx[n_train:]])
    valid_y_2 = np.array([train_y_2[s] for s in sidx[n_train:]])
    valid_y_3 = np.array([train_y_3[s] for s in sidx[n_train:]])
    train_desc = [train_desc[s] for s in sidx[:n_train]]
    train_brands = [train_brands[s] for s in sidx[:n_train]]
    train_y_1 = np.array([train_y_1[s] for s in sidx[:n_train]])
    train_y_2 = np.array([train_y_2[s] for s in sidx[:n_train]])
    train_y_3 = np.array([train_y_3[s] for s in sidx[:n_train]])

    train = (train_desc, train_brands,train_y_1,train_y_2, train_y_3)
    valid = (valid_desc, valid_brands, valid_y_1, valid_y_2, valid_y_3)
    # sample from test if test_size > 0
    if test_size > 0:
        # The test set is sorted by size, but we want to keep random
        # size example.  So we must select a random selection of the
        # examples.
        idx = np.arange(len(test[0]))
        #np.random.seed(1555)
        np.random.shuffle(idx)

        idx = idx[:test_size]
        test = ([test[0][n] for n in idx], [test[1][n] for n in idx],
                np.array([test[2][n] for n in idx]), np.array([test[3][n] for n in idx]),
                np.array([test[4][n] for n in idx]))

    if train_size > 0:
        # The test set is sorted by size, but we want to keep random
        # size example.  So we must select a random selection of the
        # examples.
        idx = np.arange(len(train[0]))
        np.random.shuffle(idx)
        idx = idx[:train_size]
        train = ([train[0][n] for n in idx], [train[1][n] for n in idx],
                [train[2][n] for n in idx],[train[3][n] for n in idx],
                [train[4][n] for n in idx])

        idx = np.arange(len(valid[0]))
        np.random.shuffle(idx)
        idx = idx[:train_size*0.1]
        valid = ([valid[0][n] for n in idx], [valid[1][n] for n in idx],
                [valid[2][n] for n in idx], [valid[3][n] for n in idx],
                [valid[4][n] for n in idx])

    data = (train, valid, test)


    return data, n_values

def merge_data(**kwargs):

    return 

# ##################### Build the neural network model #######################
# This script supports three types of models. For each one, we define a
# function that takes a Theano variable representing the input and returns
# the output layer of a neural network model built in Lasagne.
def build_mlp(input_var=None, layer_shape = None, num_units = None):
    # This creates an MLP of two hidden layers of 800 units each, followed by
    # a softmax output layer of 10 units. It applies 20% dropout to the input
    # data and 50% dropout to the hidden layers.

    # Input layer, specifying the expected input shape of the network
    # (unspecified batchsize, 1 channel, 28 rows and 28 columns) and
    # linking it to the given Theano variable `input_var`, if any:
    l_in = lasagne.layers.InputLayer(shape=(None, layer_shape),
                                     input_var=input_var)

    # Apply 20% dropout to the input data:
    l_in_drop = lasagne.layers.DropoutLayer(l_in, p=numpy.float32(0.2))

    # Add a fully-connected layer of 800 units, using the linear rectifier, and
    # initializing weights with Glorot's scheme (which is the default anyway):
    l_hid1 = lasagne.layers.DenseLayer(
            l_in_drop, num_units=800,
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.GlorotUniform())

    # We'll now add dropout of 50%:
    l_hid1_drop = lasagne.layers.DropoutLayer(l_hid1, p=0.5)

    # Another 800-unit layer:
    l_hid2 = lasagne.layers.DenseLayer(
            l_hid1_drop, num_units=800,
            nonlinearity=lasagne.nonlinearities.rectify)

    # 50% dropout again:
    l_hid2_drop = lasagne.layers.DropoutLayer(l_hid2, p=0.5)

    # Finally, we'll add the fully-connected output layer, of 10 softmax units:
    l_out = lasagne.layers.DenseLayer(
            l_hid2_drop, num_units=num_units,
            nonlinearity=lasagne.nonlinearities.softmax)

    # Each layer is linked to its incoming layer(s), so we only need to pass
    # the output layer to give access to a network in Lasagne:
    return l_out

def build_custom_mlp(input_var=None, depth=10, width=256, drop_input=np.float32(.2),
    drop_hidden=np.float32(.5), layer_shape = None, num_units = None):
    # By default, this creates the same network as `build_mlp`, but it can be
    # customized with respect to the number and size of hidden layers. This
    # mostly showcases how creating a network in Python code can be a lot more
    # flexible than a configuration file. Note that to make the code easier,
    # all the layers are just called `network` -- there is no need to give them
    # different names if all we return is the last one we created anyway; we
    # just used different names above for clarity.

    # Input layer and dropout (with shortcut `dropout` for `DropoutLayer`):

    network = lasagne.layers.InputLayer(shape=(None, layer_shape),
                                        input_var=input_var)

    #network = lasagne.layers.EmbeddingLayer(network, input_size=layer_shape, output_size=256)
    # Hidden layers and dropout:
    nonlin = lasagne.nonlinearities.rectify
    for i in range(depth):
        network = lasagne.layers.DenseLayer(
                network, width, nonlinearity=nonlin, name='hid_%d' % i)
        if drop_hidden:
            network = lasagne.layers.dropout(network, p=drop_hidden)
    # Output layer:
    softmax = lasagne.nonlinearities.softmax
    if type(num_units) == list:
        networks = []
        for idx, num_unit in enumerate(num_units):
            networks.append(lasagne.layers.DenseLayer(network, num_unit[0], nonlinearity=softmax))
    else:
        networks = lasagne.layers.DenseLayer(network, num_units, nonlinearity=softmax)
    return networks

def classifier_layer(network, prev_cat, num_units, layer_shape = None):
    #This takes in a pretrained network and concatenates it with additional
    #features. Then the combined layer is fed into the output layer

    #Output layer
    softmax = lasagne.nonlinearities.softmax

    #get all layers from the pretrained network and gets the last layer before the output layer
    layers = lasagne.layers.get_all_layers(network)
    output = lasagne.layers.get_output(layers[-2])

    #concatenate the output with additional layers
    network = T.concatenate([output, prev_cat], axis = 1)

    #use combined layer as input layer for new network and feed into output layer
    network = lasagne.layers.InputLayer(shape=(None, layer_shape),input_var=network)
    network = lasagne.layers.DenseLayer(network, num_units, nonlinearity=softmax)

    return network


# ############################# Batch iterator ###############################
# This is just a simple helper function iterating over training data in
# mini-batches of a particular size, optionally in random order. It assumes
# data is available as numpy arrays. For big datasets, you could load numpy
# arrays as memory-mapped files (np.load(..., mmap_mode='r')), or write your
# own custom data iteration function. For small datasets, you can also copy
# them to GPU at once for slightly improved performance. This would involve
# several changes in the main program, though, and is not demonstrated here.

def iterate_minibatches(inputs, target, batchsize, shuffle=False):
    assert len(inputs) == len(target)
    batches = []
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = np.arange(start_idx, start_idx + batchsize)
        batches.append(excerpt)

    return batches


# ############################## Main program ################################
# Everything else will be handled in our main program now. We could pull out
# more functions to better separate the code, but it wouldn't make it any
# easier to read.

def train_model(model='custom_mlp: ',
    mode = 'simple', #select either simple or dependent models 
    data = None,
    n_values = None,
    num_epochs=5,
    desc_n_values = 5000,
    depth = 3,
    width = 256,
    drop_in = 0.5,
    drop_hid = 0.5,
    batch_size = 32,
    learning_rate = 0.01,
    valid_freq = 100,
    save_path = '../results/',
    saveto = 'test_mlp.npz',
    reload_model = None,
    shared_params = None,
    cat = 1,
    prev_predictions = None):

    '''
    args:
        model: 'mlp', custom_mlp:','classifier_layer', or forthcoming 'image_model','multi-modal'
        data: data
        n_values: number of values ?
        num_epochs: number of epochs before quitting
        desc_n_values: 
        depth: how many layers in network
        width: units in each hidden layer
        drop_in: dropout rate for input
        drop_hid: dropout rate in hidden layers
        batch_size: how many to run at a time
        learning_rate: learning rate of SGD
        valid_freq: how often to validate
        save_path: where to save the resulting model
        saveto: name of the file where saving
        reload_model = None,
        shared_params = None,
        cat = 1,
        prev_predictions = None):
    '''

    train, valid, test = data

    layer_shape = desc_n_values + n_values['brands'] #CG TODO: add image dimensions here

    # Prepare Theano variables for inputs and target
    input_var = T.matrix('inputs',dtype='int64')
    target_var = T.ivector('target')

    n_val_keys = n_values.keys()
    if cat != 1:
        prev_cat_var = T.matrix('prev_inputs',dtype='int64')
        classifier_layer_shape = width + n_values[n_val_keys[cat]]
        train_prev_cat = train[cat]
        valid_prev_cat = valid[cat]
        test_prev_cat = prev_predictions

    # Create neural network model (depending on first command line parameter)
    start_time = time.time()
    print("Building model and compiling functions...")
    if model == 'mlp':
        network = build_mlp(input_var, layer_shape, n_values['y_1'])
    elif model == 'custom_mlp':
        #depth, width, drop_in, drop_hid = model.split(':', 1)[1].split(',')
        #network = build_custom_mlp(input_var, int(depth), int(width),
        #                          float(drop_in), float(drop_hid))
        network = build_custom_mlp(input_var, depth, width, drop_in, drop_hid, layer_shape, n_values['y_1'])
    elif model == 'classifier_layer':
        network = build_custom_mlp(input_var, depth, width, drop_in, drop_hid, layer_shape, n_values['y_1'])
        if reload_model is not None:
            with np.load(reload_model) as f:
                param_values = [f['arr_%d' % i] for i in range(len(f.files)-7)]
            lasagne.layers.set_all_param_values(network, param_values)
        if shared_params is not None:
            lasagne.layers.set_all_param_values(network, shared_params)
        network = classifier_layer(network, 
            prev_cat_var, n_values[n_val_keys[cat]], layer_shape = classifier_layer_shape)
    else:
        print("Unrecognized model type %r." % model)
        return

    # Create a loss expression for training, i.e., a scalar objective we want
    # to minimize (for our multi-class problem, it is the cross-entropy loss):
    prediction = lasagne.layers.get_output(network)
    loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
    loss = loss.mean()


    # We could add some weight decay as well here, see lasagne.regularization.

    # Create update expressions for training, i.e., how to modify the
    # parameters at each training step. Here, we'll use adadelta,
    # but Lasagne offers plenty more.
    params = lasagne.layers.get_all_params(network, trainable=True)
    updates = lasagne.updates.adadelta(
            loss, params, learning_rate=0.01)

    # Create a loss expression for validation/testing. The crucial difference
    # here is that we do a deterministic forward pass through the network,
    # disabling dropout layers.
    test_prediction = lasagne.layers.get_output(network, deterministic=True)
    test_loss = lasagne.objectives.categorical_crossentropy(test_prediction,
                                                            target_var)
    test_loss = test_loss.mean()

    #TODO: three target layers, loss is the sum of the three.  

    # As a bonus, also create an expression for the classification accuracy:
    # TODO: separate accuracy for the three
    test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var),
                      dtype=theano.config.floatX)

    # Compile a function performing a training step on a mini-batch (by giving
    # the updates dictionary) and returning the corresponding training loss:
    if cat != 1:
        train_fn = theano.function([input_var, prev_cat_var, target_var], loss, updates=updates)
        # Compile a second function computing the validation loss and accuracy:
        val_fn = theano.function([input_var, prev_cat_var, target_var], [test_loss, test_acc])
        preds = theano.function([input_var, prev_cat_var],test_prediction)
    else:
        train_fn = theano.function([input_var, target_var], loss, updates=updates)
        val_fn = theano.function([input_var, target_var], [test_loss, test_acc])
        preds = theano.function([input_var],test_prediction)

    history_train_errs = []
    history_valid_errs = []
    # Finally, launch the training loop.
    print("Starting training...")
    # We iterate over epochs:
    for epoch in range(num_epochs):
        # In each epoch, we do a full pass over the training data:
        train_err = 0
        train_batches = 0
        start_time = time.time()
        t_idx = 0
        for batch in iterate_minibatches(train[0], train[1], batch_size, shuffle=True):
            t_idx += 1
            inputs = [[train[0][idx] for idx in batch],[train[1][idx] for idx in batch]]
            target = [train[2][idx] for idx in batch]
            if cat != 1:
                prev_inputs = [train_prev_cat[idx] for idx in batch]
                print(len(prev_inputs))
                target = [train[cat + 1][idx] for idx in batch]
                prev_inputs = one_hot_encode_features(prev_inputs,
                    n_values = n_values[n_val_keys[cat]])
                print(prev_inputs.shape)
            desc = one_hot_encode_features(inputs[0],n_values = n_values['desc'])
            brands = one_hot_encode_features(inputs[1],n_values = n_values['brands'])
            inputs = np.hstack((desc, brands))
            if cat != 1:
                train_err += train_fn(inputs, prev_inputs, target)
            else:
                train_err += train_fn(inputs, target)
            train_batches += 1

            if t_idx % valid_freq == 0:
                if cat != 1:
                    err, acc = val_fn(inputs, prev_inputs, target)
                else:
                    err, acc = val_fn(inputs,target)
                history_train_errs.append([err, acc])
                np.savez(save_path + saveto,
                        history_train_errs=history_train_errs,
                        history_valid_errs = history_valid_errs,
                        layers = lasagne.layers.get_all_layers(network),
                         *lasagne.layers.get_all_param_values(network))

        # And a full pass over the validation data:
        val_err = 0
        val_acc = 0
        val_batches = 0
        for batch in iterate_minibatches(valid[0],valid[1], batch_size, shuffle=False):
            inputs = [[valid[0][idx] for idx in batch],[valid[1][idx] for idx in batch]]
            target = [valid[2][idx] for idx in batch]
            #add image vectors here
            if cat != 1:
                prev_inputs = [valid_prev_cat[idx] for idx in batch]
                target = [valid[cat + 1][idx] for idx in batch]
                prev_inputs = one_hot_encode_features(prev_inputs,
                    n_values = n_values[n_val_keys[cat]])
            desc = one_hot_encode_features(inputs[0],n_values = n_values['desc'])
            brands = one_hot_encode_features(inputs[1],n_values = n_values['brands'])
            inputs = np.hstack((desc, brands)) #hstack image vectors
            if cat != 1:
                err, acc = val_fn(inputs, prev_inputs, target)
            else:
                err, acc = val_fn(inputs, target)
            val_err += err
            val_acc += acc
            val_batches += 1

            if t_idx % valid_freq == 0:
                if cat != 1:
                    err, acc = val_fn(inputs, prev_inputs, target)
                else:
                    err, acc = val_fn(inputs, target)
                history_train_errs.append([err, acc])
                print('saving...')
                np.savez(save_path + saveto,
                        history_train_errs=history_train_errs,
                        history_valid_errs = history_valid_errs,
                         *lasagne.layers.get_all_param_values(network))

        # Then we print the results for this epoch:
        print("Epoch {} of {} took {:.3f}s".format(
            epoch + 1, num_epochs, time.time() - start_time))
        print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
        print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))
        print("  validation accuracy:\t\t{:.2f} %".format(
            val_acc / val_batches * 100))


    end_time = time.time()
    print("The code ran for %d epochs, with %f sec/epochs" % (
        (num_epochs), (end_time - start_time) / (1. * (num_epochs))))

    # After training, we compute and print the test error:
    test_err = 0
    test_acc = 0
    test_batches = 0
    test_preds = np.zeros(len(test[0]))
    for batch in iterate_minibatches(test[0],test[2], batch_size, shuffle=False):
        inputs = [[test[0][idx] for idx in batch],[test[1][idx] for idx in batch]]
        target = [test[2][idx] for idx in batch]
        desc = one_hot_encode_features(inputs[0],n_values = n_values['desc'])
        brands = one_hot_encode_features(inputs[1],n_values = n_values['brands'])
        if cat != 1:
                prev_inputs = [test_prev_cat[idx] for idx in batch]
                target = [test[cat + 1][idx] for idx in batch]
                prev_inputs = one_hot_encode_features(prev_inputs,
                    n_values = n_values[n_val_keys[cat]])
        inputs = np.hstack((desc, brands))
        if cat != 1:
            err, acc = val_fn(inputs, prev_inputs, target)
            pred_prob = preds(inputs, prev_inputs)
        else:
            err, acc = val_fn(inputs, target)
            pred_prob = preds(inputs)
        pred = pred_prob.argmax(axis = 1)
        test_preds[batch[0]:batch[-1]+1] = pred
        test_err += err
        test_acc += acc
        test_batches += 1
    print("Final results:")
    print("  test loss:\t\t\t{:.6f}".format(test_err / test_batches))
    print("  test accuracy:\t\t{:.2f} %".format(
        test_acc / test_batches * 100))

    # Optionally, you could now dump the network weights to a file like this:
    np.savez(save_path + saveto,train_err=train_err / train_batches,
                        valid_err=val_err / val_batches, 
                        test_err=test_err / test_batches,
                        history_train_errs=history_train_errs,
                        history_valid_errs = history_valid_errs,
                        layers = lasagne.layers.get_all_layers(network),
                        predictions = test_preds,
                         *lasagne.layers.get_all_param_values(network))

    param_values = lasagne.layers.get_all_param_values(network)
    #
    # And load them again later on like this:
    # with np.load('model.npz') as f:
    #     param_values = [f['arr_%d' % i] for i in range(len(f.files))]
    # lasagne.layers.set_all_param_values(network, param_values)

    return param_values, test_preds

def get_all_params(network):
    params = []
    for n in network:
        p = []
        p.append(lasagne.layers.get_all_param_values(n))
        params.append(p)

    return params


def train_simple_model(model='custom_mlp', 
    data = None,
    n_values = None,
    num_epochs=5,
    desc_n_values = 5000,
    depth = 10,
    width = 256,
    drop_in = 0.,
    drop_hid = 0.,
    batch_size = 32,
    learning_rate = 0.01,
    valid_freq = 100,
    save_path = '../results/',
    saveto = 'test_mlp.npz',
    reload_model = None,
    num_targets = 1):

    train, valid, test = data

    layer_shape = desc_n_values + n_values['brands'] #CG TODO: add image dimensions here

    # Prepare Theano variables for inputs and target
    input_var = T.matrix('inputs',dtype='float32')

    target_var = []
    for i in range(num_targets):
        target_var.append(T.vector('target_%s' % i,dtype = 'int64'))

    n_val_keys = n_values.keys()

    # Create neural network model (depending on first command line parameter)
    start_time = time.time()
    print("Building model and compiling functions...")
    if model == 'mlp':
        network = build_mlp(input_var, layer_shape, n_values['y_1'])
    elif model == 'custom_mlp':
        #depth, width, drop_in, drop_hid = model.split(':', 1)[1].split(',')
        #network = build_custom_mlp(input_var, int(depth), int(width),
        #                          float(drop_in), float(drop_hid))
        network = build_custom_mlp(input_var, 
            depth, width, drop_in, drop_hid, 
            layer_shape, [[n_values['y_1']],[n_values['y_2']],[n_values['y_3']]])
    else:
        print("Unrecognized model type %r." % model)
        return

    # Create a loss expression for training, i.e., a scalar objective we want
    # to minimize (for our multi-class problem, it is the cross-entropy loss):
    prediction = []
    for n in network:
        prediction.append(lasagne.layers.get_output(n))

    #for p,t in zip(prediction,target_var):
    #    loss += lasagne.objectives.categorical_crossentropy(p, t)
    
    pred_concat = T.concatenate(prediction, axis = 1)  
    target_concat = T.concatenate(target_var)
    loss = lasagne.objectives.categorical_crossentropy(prediction[0],target_var[0]) + lasagne.objectives.categorical_crossentropy(prediction[1],target_var[1]) + lasagne.objectives.categorical_crossentropy(prediction[2],target_var[2])
    loss = loss.mean()


    # We could add some weight decay as well here, see lasagne.regularization.

    # Create update expressions for training, i.e., how to modify the
    # parameters at each training step. Here, we'll use adadelta,
    params = []
    for i, n in enumerate(network):
        if i == 0:
            p = lasagne.layers.get_all_params(n, trainable=True)
        else:
            p = lasagne.layers.get_all_params(n, trainable=True)[-2:]
        params += p
    updates = lasagne.updates.adadelta(
            loss, params, learning_rate=0.01)

    # Create a loss expression for validation/testing. The crucial difference
    # here is that we do a deterministic forward pass through the network,
    # disabling dropout layers.
    test_prediction = []
    test_loss = []
    test_acc = []
    preds = []
    for n,t in zip(network,target_var):
        p = lasagne.layers.get_output(n, deterministic=True)
        test_prediction.append(p)
        l = lasagne.objectives.categorical_crossentropy(p,t)
        test_loss.append(l.mean())

        # As a bonus, also create an expression for the classification accuracy:
        acc = T.mean(T.eq(T.argmax(p, axis=1), t),
                      dtype=theano.config.floatX)
        test_acc.append(acc)
        preds.append(theano.function([input_var],p))


    inputs = []
  
    val_fn = []
    train_fn = theano.function([input_var] + target_var, loss, updates=updates)
    for t,l,a in zip(target_var, test_loss, test_acc):
        val_fn.append(theano.function([input_var, t], [l, a]))

    history_train_errs = []
    history_valid_errs = []
    # Finally, launch the training loop.
    print("Starting training...")
    # We iterate over epochs:
    for epoch in range(num_epochs):
        # In each epoch, we do a full pass over the training data:
        train_err = 0
        train_batches = 0
        start_time = time.time()
        t_idx = 0
        for batch in iterate_minibatches(train[0], train[1], batch_size, shuffle=True):
            t_idx += 1
            inputs = [[train[0][idx] for idx in batch],[train[1][idx] for idx in batch]]
            target = [[train[2][idx] for idx in batch],[train[3][idx] for idx in batch],
                        [train[4][idx] for idx in batch]]
            #CG TODO: iterate over image vectors
            desc = one_hot_encode_features(inputs[0],n_values = n_values['desc'])
            brands = one_hot_encode_features(inputs[1],n_values = n_values['brands'])
            inputs = np.hstack((desc, brands)) #CG TODO: hstack with images as well
            train_err += train_fn(inputs, target[0], target[1], target[2])
            train_batches += 1

            if t_idx % valid_freq == 0:
                err = []
                acc = []
                for i in range(num_targets):
                    e, a = val_fn[i](inputs,target[i])
                    err.append(e)
                    acc.append(a)
                history_train_errs.append([err, acc])
                np.savez(save_path + saveto,
                        history_train_errs=history_train_errs,
                        history_valid_errs = history_valid_errs,
                        *params)

        # And a full pass over the validation data:
        val_err = np.zeros(num_targets)
        val_acc = np.zeros(num_targets)
        val_batches = 0
        for batch in iterate_minibatches(valid[0],valid[1], batch_size, shuffle=False):
            inputs = [[valid[0][idx] for idx in batch],[valid[1][idx] for idx in batch]]
            target = [[valid[2][idx] for idx in batch], [valid[3][idx] for idx in batch],
                        [valid[4][idx] for idx in batch]]

            desc = one_hot_encode_features(inputs[0],n_values = n_values['desc'])
            brands = one_hot_encode_features(inputs[1],n_values = n_values['brands'])
            inputs = np.hstack((desc, brands))

            for i in range(num_targets):
                e,a = val_fn[i](inputs, target[i])
                val_err[i] += e
                val_acc[i] += a
            val_batches += 1

            params = get_all_params(network)

            if t_idx % valid_freq == 0:
                err = []
                acc = []
                for i in range(num_targets):
                    e,a = val_fn[i](inputs, target[i])
                    err.append(e)
                    acc.append(a)
                history_train_errs.append([err, acc])
                print('saving...')
                np.savez(save_path + saveto,
                        history_train_errs=history_train_errs,
                        history_valid_errs = history_valid_errs,
                         *params)

        # Then we print the results for this epoch:
        print("Epoch {} of {} took {:.3f}s".format(
            epoch + 1, num_epochs, time.time() - start_time))
        max_train = np.max(train_err / train_batches)
        min_train = np.min(train_err / train_batches)
        max_val = np.max(val_err / val_batches)
        min_val = np.min(val_err / val_batches)
        avg_val_acc = np.mean(val_acc / val_batches)
        print("  max training loss:\t\t{:.6f}".format(max_train))
        print("  min training loss:\t\t{:.6f}".format(min_train))
        print("  max validation loss:\t\t{:.6f}".format(max_val))
        print("  min validation loss:\t\t{:.6f}".format(min_val))
        print("  avg validation accuracy:\t\t{:.2f} %".format(
            avg_val_acc * 100))


    end_time = time.time()
    print("The code ran for %d epochs, with %f sec/epochs" % (
        (num_epochs), (end_time - start_time) / (1. * (num_epochs))))

    # After training, we compute and print the test error:
    test_err = np.zeros(num_targets)
    test_acc = np.zeros(num_targets)
    test_batches = 0
    test_preds = []
    for i in range(num_targets):
        test_preds.append(np.zeros(len(test[i+2])))
    for batch in iterate_minibatches(test[0],test[2], batch_size, shuffle=False):
        inputs = [[test[0][idx] for idx in batch],[test[1][idx] for idx in batch]]
        target = [[test[2][idx] for idx in batch],[test[3][idx] for idx in batch],
                    [test[4][idx] for idx in batch]]
        desc = one_hot_encode_features(inputs[0],n_values = n_values['desc'])
        brands = one_hot_encode_features(inputs[1],n_values = n_values['brands'])
        inputs = np.hstack((desc, brands))
        
        for i in range(num_targets):
            e,a = val_fn[i](inputs, target[i])
            pred_prob = preds[i](inputs)
            pred = pred_prob.argmax(axis = 1)
            test_preds[i][batch[0]:batch[-1]+1] = pred
            test_err[i] += e
            test_acc[i] += a
        test_batches += 1

    max_err = np.max(test_err / test_batches)
    min_err = np.min(test_err / test_batches)
    avg_acc = np.mean(test_acc / test_batches)
    max_acc = np.max(test_acc / test_batches)
    min_acc = np.min(test_acc / test_batches)
    print("Final results:")
    print("  max test loss:\t\t\t{:.6f}".format(max_err))
    print("  min test loss:\t\t\t{:.6f}".format(min_err))
    print("  avg test accuracy:\t\t{:.2f} %".format(
        avg_acc * 100))
    print("  max test accuracy:\t\t{:.2f} %".format(
        max_acc * 100))
    print("  min test accuracy:\t\t{:.2f} %".format(
        min_acc * 100))

    params = get_all_params(network)
    # Optionally, you could now dump the network weights to a file like this:
    np.savez(save_path + saveto,train_err=train_err / train_batches,
                        valid_err=val_err / val_batches, 
                        test_err=test_err / test_batches,
                        history_train_errs=history_train_errs,
                        history_valid_errs = history_valid_errs,
                        predictions = test_preds,
                         *params)

    #
    # And load them again later on like this:
    # with np.load('model.npz') as f:
    #     param_values = [f['arr_%d' % i] for i in range(len(f.files))]
    # lasagne.layers.set_all_param_values(network, param_values)

    return params, test_preds

def main():
    '''
    set parameters,load data and train model
    '''

    #set variable values that will be used by all models
    desc_n_values = 5000
    epochs = 5
    depth = 10
    width = 256
    save_path = '../results/simple_mlp/'

    # Load the dataset
    print("Loading data...")
    data, n_values = get_data(
        test_size=100,  # If >0, we keep only this number of test example.
        train_size = 1000, # If >0, we keep only this number of train example.
        valid_portion = 0.1,
        desc_n_values = desc_n_values)

    #create model
    
    params, preds = train_simple_model(model='custom_mlp', 
        data = data,
        n_values = n_values,
        num_epochs=epochs,
        desc_n_values = desc_n_values,
        depth = depth,
        width = width,
        batch_size = 32,
        learning_rate = 0.01,
        valid_freq = 100,
        save_path = save_path,
        saveto = 'simple_mlp.npz',
        reload_model = None,
        num_targets = 3)
    

    '''
    print('train level 1')
    param_values_1, test_preds_1 = train_model(model='custom_mlp',
        data = data,
        n_values = n_values,
        num_epochs=epochs,
        desc_n_values = desc_n_values,
        depth = depth,
        width = width,
        drop_in = 0.2,
        drop_hid = 0.5,
        batch_size = 32,
        learning_rate = 0.01,
        valid_freq = 100,
        save_path = save_path,
        saveto = 'mlp_cat_1.npz',
        reload_model = None,
        shared_params = None,
        cat = 1,
        prev_predictions = None)

    print('train level 2')
    param_values_2, test_preds_2 = train_model(model='classifier_layer',
        data = data,
        n_values = n_values,
        num_epochs=epochs,
        desc_n_values = desc_n_values,
        depth = depth,
        width = width,
        drop_in = 0.2,
        drop_hid = 0.5,
        batch_size = 32,
        learning_rate = 0.01,
        valid_freq = 100,
        save_path = save_path,
        saveto = 'mlp_cat_2.npz',
        reload_model = None,
        shared_params = param_values_1,
        cat = 2,
        prev_predictions = test_preds_2)

    print('train level 3')
    param_values_3, test_preds_3 = train_model(model='classifier_layer',
        data = data,
        n_values = n_values,
        num_epochs=epochs,
        desc_n_values = desc_n_values,
        depth = depth,
        width = width,
        drop_in = 0.2,
        drop_hid = 0.5,
        batch_size = 32,
        learning_rate = 0.01,
        valid_freq = 100,
        save_path = save_path,
        saveto = 'mlp_cat_3.npz',
        reload_model = None,
        shared_params = param_values_2,
        cat = 3,
        prev_predictions = test_preds_2)

    np.savez(save_path + 'targets.pkl', data[2])
    '''

if __name__ == '__main__':
    if ('--help' in sys.argv) or ('-h' in sys.argv):
        print("Trains a neural network on MNIST using Lasagne.")
        print("Usage: %s [MODEL [EPOCHS]]" % sys.argv[0])
        print()
        print("MODEL: 'mlp' for a simple Multi-Layer Perceptron (MLP),")
        print("       'custom_mlp:DEPTH,WIDTH,DROP_IN,DROP_HID' for an MLP")
        print("       with DEPTH hidden layers of WIDTH units, DROP_IN")
        print("       input dropout and DROP_HID hidden dropout,")
        print("       'cnn' for a simple Convolutional Neural Network (CNN).")
        print("EPOCHS: number of training epochs to perform (default: 500)")
    else:
        kwargs = {}
        if len(sys.argv) > 1:
            kwargs['model'] = sys.argv[1]
        if len(sys.argv) > 2:
            kwargs['num_epochs'] = int(sys.argv[2])
        main(**kwargs)
