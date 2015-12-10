"""
Adapted from the MNIST example codes in Lasagne

"""
#TODO:
#in iterate_minibatch: turn off shuffle.
#get rid of anything that says one-hot encode
#delete the vanilla build_mlp


from __future__ import print_function
from utils import create_log,plog,fplog

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

def build_custom_mlp(input_var=None, depth=10, width=256, drop_input=np.float32(.2),
    drop_hidden=np.float32(.5), layer_shape = None, num_units = None):
    '''
    args:
        layer_shape: width of input X
        drop_input: fraction to dropout from input layer
        drop_hidden: fraction to dropout from hidden layers
        num_units: List or int. 19, 40, or 240ish, or [19,40,240ish].  number of choices in each category.  
        input_var: input variable.  matrix
    '''
    # By default, this creates the same network as `build_mlp`, but it can be
    # customized with respect to the number and size of hidden layers. This
    # mostly showcases how creating a  network in Python code can be a lot more
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
        for num_unit in num_units:
            networks.append(lasagne.layers.DenseLayer(network, num_unit[0], nonlinearity=softmax))
    else:
        networks = lasagne.layers.DenseLayer(network, num_units, nonlinearity=softmax)
    return networks


# ############################# Batch iterator ###############################
# This is just a simple helper function iterating over training data in
# mini-batches of a particular size, optionally in random order. It assumes
# data is available as numpy arrays. For big datasets, you could load numpy
# arrays as memory-mapped files (np.load(..., mmap_mode='r')), or write your
# own custom data iteration function. For small datasets, you can also copy
# them to GPU at once for slightly improved performance. This would involve
# several changes in the main program, though, and is not demonstrated here.
def iterate_minibatches(data, batchsize, shuffle=False):
    inputs = data[0]
    targets = data[1:4]

    assert len(inputs) == len(targets[0])
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], [y[excerpt] for y in targets]

def get_all_params(network):
    params = []
    for n in network:
        p = []
        p.append(lasagne.layers.get_all_param_values(n))
        params.append(p)

    return params

#CG Use this model.  Must be all 3 right now. 
def train_simple_model(data = None,
    n_values = None,
    num_epochs=5,
    depth = 10,
    width = 256,
    drop_in = 0.,
    drop_hid = 0.,
    batch_size = 32,
    learning_rate = 0.01,
    valid_freq = 100,
    save_path = '../results/',
    options_dict = None,
    reload_model = None,
    num_targets = 3):

    train, valid, test = data

    #X width, so the model knows how wide to make the first layer
    layer_shape = train[0].shape[1] 

    # Prepare Theano variables for inputs and target
    input_var = T.matrix('inputs',dtype='float32')

    #CG: num_targets is 1 or 3.  int64 fine because it goes into output
    target_var = []
    for i in range(num_targets):
        target_var.append(T.vector('target_%s' % i,dtype = 'int32'))

    # Create neural network model (depending on first command line parameter)
    #CG: ignore mlp, maybe remove this whole switch.
    #CG 1: build network
    start_time = time.time()
    fplog("Building model and compiling functions...")
    network = build_custom_mlp(input_var, 
        depth, width, drop_in, drop_hid, 
        layer_shape, [[n_values['y_1']],[n_values['y_2']],[n_values['y_3']]])


    # Create a loss expression for training, i.e., a scalar objective we want
    # to minimize (for our multi-class problem, it is the cross-entropy loss):
    # CG 2: Make prediction
    prediction = []
    for n in network:
        prediction.append(lasagne.layers.get_output(n))

    loss=0

    #for p,t in zip(prediction,target_var):
    #    loss += lasagne.objectives.categorical_crossentropy(p, t)
    loss = lasagne.objectives.categorical_crossentropy(prediction[0],target_var[0]) + lasagne.objectives.categorical_crossentropy(prediction[1],target_var[1]) + lasagne.objectives.categorical_crossentropy(prediction[2],target_var[2])
    loss=loss.mean()



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


  
    val_fn = []
    train_fn = theano.function([input_var] + target_var, loss, updates=updates)
    for t,l,a in zip(target_var, test_loss, test_acc):
        val_fn.append(theano.function([input_var, t], [l, a]))

    history_train_errs = []
    history_valid_errs = []
    # Finally, launch the training loop.
    fplog("Starting training...")
    # We iterate over epochs:
    for epoch in range(num_epochs):
        # In each epoch, we do a full pass over the training data:
        train_err = 0
        train_batches = 0
        start_time = time.time()

        for batch in iterate_minibatches(train, batch_size, shuffle=False):
            inputs, targets = batch
            train_err += train_fn(inputs, targets[0], targets[1], targets[2])
            train_batches += 1

            if train_batches % valid_freq == 0:
                err = []
                acc = []
                for i in range(num_targets):
                    e, a = val_fn[i](inputs,targets[i])
                    err.append(e)
                    acc.append(a)
                history_train_errs.append([err, acc])
                save_to_results_file(var_string,results_path)
                np.savez(save_path,
                        history_train_errs=history_train_errs,
                        history_valid_errs = history_valid_errs,
                        options_dict=options_dict,
                        *params)

        # And a full pass over the validation data:
        #CG: Repeat some of the above, or make it a function
        val_err = np.zeros(num_targets)
        val_acc = np.zeros(num_targets)
        val_batches = 0
        for batch in iterate_minibatches(valid, batch_size, shuffle=False):
            inputs, targets = batch

            #calculate error and accuracy separately for each target
            for i in range(num_targets):
                e,a = val_fn[i](inputs, targets[i])
                val_err[i] += e
                val_acc[i] += a
            val_batches += 1

            params = get_all_params(network)

            if train_batches % valid_freq == 0:
                err = []
                acc = []
                for i in range(num_targets):
                    e,a = val_fn[i](inputs, targets[i])
                    err.append(e)
                    acc.append(a)
                history_train_errs.append([err, acc])
                fplog('saving...')
                np.savez(save_path,
                        history_train_errs=history_train_errs,
                        history_valid_errs = history_valid_errs,
                        options_dict=options_dict,
                         *params)

        # Then we fplog the results for this epoch:
        fplog("Epoch {} of {} took {:.3f}s".format(
            epoch + 1, num_epochs, time.time() - start_time))
        max_train = np.max(train_err / train_batches)
        min_train = np.min(train_err / train_batches)
        max_val = np.max(val_err / val_batches)
        min_val = np.min(val_err / val_batches)
        avg_val_acc = np.mean(val_acc / val_batches)
        fplog("  max training loss:\t\t{:.6f}".format(max_train))
        fplog("  min training loss:\t\t{:.6f}".format(min_train))
        fplog("  max validation loss:\t\t{:.6f}".format(max_val))
        fplog("  min validation loss:\t\t{:.6f}".format(min_val))
        fplog("  avg validation accuracy:\t\t{:.2f} %".format(
            avg_val_acc * 100))


    end_time = time.time()
    fplog("The code ran for %d epochs, with %f sec/epochs" % (
        (num_epochs), (end_time - start_time) / (1. * (num_epochs))))

    # After training, we compute and fplog the test error:
    test_err = np.zeros(num_targets)
    test_acc = np.zeros(num_targets)
    test_batches = 0
    test_preds = []
    y_test_list = test[1:]
    for i in range(num_targets):
        test_preds.append(np.zeros(len(y_test_list[i])))
    for batch in iterate_minibatches(train, batch_size, shuffle=False):
        inputs, targets = batch
        
        for i in range(num_targets):
            e,a = val_fn[i](inputs, targets[i])
            pred_prob = preds[i](inputs)
            pred = pred_prob.argmax(axis = 1)
            test_preds[i] = np.append(test_preds[i],pred)
            test_err[i] += e
            test_acc[i] += a
        test_batches += 1

    max_err = np.max(test_err / test_batches)
    min_err = np.min(test_err / test_batches)
    avg_acc = np.mean(test_acc / test_batches)
    max_acc = np.max(test_acc / test_batches)
    min_acc = np.min(test_acc / test_batches)
    fplog("Final results:")
    fplog("  max test loss:\t\t\t{:.6f}".format(max_err))
    fplog("  min test loss:\t\t\t{:.6f}".format(min_err))
    fplog("  avg test accuracy:\t\t{:.2f} %".format(
        avg_acc * 100))
    fplog("  max test accuracy:\t\t{:.2f} %".format(
        max_acc * 100))
    fplog("  min test accuracy:\t\t{:.2f} %".format(
        min_acc * 100))

    params = get_all_params(network)


    # Optionally, you could now dump the network weights to a file like this:
    np.savez(save_path, train_err=train_err / train_batches,
                        valid_err=val_err / val_batches, 
                        test_err=test_err / test_batches,
                        history_train_errs=history_train_errs,
                        history_valid_errs = history_valid_errs,
                        predictions = test_preds,
                        options_dict=options_dict,
                         *params)

    #
    # And load them again later on like this:
    # with np.load('model.npz') as f:
    #     param_values = [f['arr_%d' % i] for i in range(len(f.files))]
    # lasagne.layers.set_all_param_values(network, param_values)

    return params, test_preds

def main(data,n_values):
    '''
    set parameters,load data and train model
    '''

    params, preds = train_simple_model(data = data,
        n_values = n_values,
        num_epochs=5,
        depth = 10,
        width = 256,
        batch_size = 32,
        learning_rate = 0.01,
        valid_freq = 100,
        save_path = '../results/simple_mlp/',
        options_dict = None,
        reload_model = None,
        num_targets = 3)

if __name__ == '__main__':
    if ('--help' in sys.argv) or ('-h' in sys.argv):
        fplog("Trains a neural network on MNIST using Lasagne.")
        fplog("Usage: %s [MODEL [EPOCHS]]" % sys.argv[0])
        fplog()
        fplog("MODEL: 'mlp' for a simple Multi-Layer Perceptron (MLP),")
        fplog("       'custom_mlp:DEPTH,WIDTH,DROP_IN,DROP_HID' for an MLP")
        fplog("       with DEPTH hidden layers of WIDTH units, DROP_IN")
        fplog("       input dropout and DROP_HID hidden dropout,")
        fplog("       'cnn' for a simple Convolutional Neural Network (CNN).")
        fplog("EPOCHS: number of training epochs to perform (default: 500)")
    else:
        kwargs = {}
        if len(sys.argv) > 1:
            kwargs['model'] = sys.argv[1]
        if len(sys.argv) > 2:
            kwargs['num_epochs'] = int(sys.argv[2])
        main(**kwargs)
