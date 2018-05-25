"""
CSCI 497E/571
Program 2 - DNN
Authors: Josh Loehr and Nicholas Majeske

A general purpose deep neural network for classification and regression.

"""
import argparse
import numpy as np
import tensorflow as tf
from datetime import datetime
from sys import stderr

# constant for adaptive gradient methods
momentum = 0.5 

# Map script arguments to tensorflow functions
hidden_acts = {'sig': tf.sigmoid, 'tanh': tf.tanh, 'relu': tf.nn.relu}
output_acts = {'C': tf.nn.softmax, 'R': tf.identity}
optimizers = {'adam': lambda lr: tf.train.AdamOptimizer(lr, momentum), 
             'momentum': lambda lr: tf.train.MomentumOptimizer(lr, momentum), 
             'grad': tf.train.GradientDescentOptimizer}
loss_funcs = {'C':tf.nn.sparse_softmax_cross_entropy_with_logits, 
             'R': tf.losses.mean_squared_error}


def parse_args():
    parser = argparse.ArgumentParser("prog2.py")
    parser.add_argument('-v', action='store_true', 
            help='operate in verbose mode')
    parser.add_argument('-train_feat', type=str, required=True, 
            help='the name of the training set feature file')
    parser.add_argument('-train_target', type=str, required=True, 
            help='the name of the training set target file')
    parser.add_argument('-dev_feat', type=str, required=True, 
            help='the name of the dev set feature file')
    parser.add_argument('-dev_target', type=str, required=True,
            help='the name of the dev set target file')
    parser.add_argument('-epochs', type=int, default=100,
            help='total number of epochs to train for')
    parser.add_argument('-learnrate', type=float, default=0.01,
            help='step size to use for training')
    parser.add_argument('-nunits', type=int, default=10,
            help='the dimension of the hidden layers')
    parser.add_argument('-type', type=str, choices=['C','R'], required=True,
            help='problem mode (C for classification; R for regression)')
    parser.add_argument('-hidden_act', type=str, choices=['sig','tanh','relu'], default='relu',
            help='nonlinearity to apply at each hidden layer')
    parser.add_argument('-optimizer', type=str, choices=['adam','momentum','grad'], default='grad',
            help='optimizer algorithm to use')
    parser.add_argument('-init_range', type=float, default=1.0,
            help='range over which to uniformly initialize parameters')
    parser.add_argument('-num_classes', type=int, default=1,
            help='number of classes (for classification only)')
    parser.add_argument('-mb', type=int, default=None,
            help='minibatch size')
    parser.add_argument('-nlayers', type=int, default=1,
            help='number of hidden layers')
    return parser.parse_args()


def read_data(features_file, targets_file, C):
    return {'features': np.loadtxt(features_file), 'targets': np.loadtxt(targets_file)}


def dnn_layer(X, shape, name, init_range, activation=None):
    with tf.name_scope(name) as scope:
        with tf.name_scope("W"):
            W = tf.Variable(tf.random_uniform(shape, minval=-init_range, maxval=init_range), name="weights")
            tf.summary.scalar('mean', tf.reduce_mean(W))
        with tf.name_scope("b"):
            b = tf.Variable(tf.zeros([shape[1]]), name="biases")
            tf.summary.scalar('mean', tf.reduce_mean(b))
        z = tf.matmul(X, W) + b
        if activation:
            return activation(z)
        else:
            return z

def build_graph(X, D, L, K, C, init_range, hidden_act, output_act):
    prev_layer = X
    for i in range(0,K):
        shape = [prev_layer.shape[1].value,L]
        prev_layer = dnn_layer(prev_layer, shape, "layer%d"%i, init_range, activation=hidden_act)

    output = dnn_layer(prev_layer, [L,C], "output", init_range)
    return output, output_act(output)

def main():
    args = parse_args()

    do_regression = args.type == 'R'
    do_classification = args.type == 'C'
    verbose = args.v

    lr = args.learnrate
    epochs = args.epochs
    init_range = args.init_range

    hidden_act = hidden_acts[args.hidden_act]
    output_act = output_acts[args.type]
    optimizer  = optimizers[args.optimizer]
    loss_func  = loss_funcs[args.type]

    # Load train and dev data
    train = read_data(args.train_feat, args.train_target, args.num_classes)
    dev = read_data(args.dev_feat, args.dev_target, args.num_classes)

    # Set up layer dimensions
    N = train['features'].shape[0]
    D = train['features'].shape[1] if len(train['features'].shape) == 2 else 1
    L = args.nunits
    K = args.nlayers


    # If regression, infer output dimension. Otherwise use num_classes.
    if do_regression:
        C = train['targets'].shape[1] if len(train['targets'].shape) == 2 else 1
    else:
        C = args.num_classes

    mb = args.mb if args.mb else N
    
    # Set up X and Y dimensions
    feature_shape = [None,D] if D > 1 else [None,]
    if do_classification or C == 1:
        target_shape = [None,]
    else:
        target_shape = [None,C]
    
    X = tf.placeholder(tf.float32, shape=feature_shape, name="X")
    Y = tf.placeholder(tf.int32, shape=target_shape, name="Y")

    # Build the DNN graph
    with tf.name_scope("dnn") as scope:
        z_out, a_out = build_graph(X, D, L, K, C, init_range, hidden_act, output_act)
        if C == 1:
            a_out = tf.squeeze(a_out)
            z_out = tf.squeeze(z_out)

    # Add the loss calculation to the graph
    with tf.name_scope("loss") as scope:
        if do_regression:
            loss = loss_func(Y, a_out)
        else:
            cross_entropy = loss_func(labels=Y, logits=z_out, name="cross_entropy")
            loss = tf.reduce_mean(cross_entropy, name="loss")
        tf.summary.scalar("loss", loss)

    # Build in the training step
    with tf.name_scope("train") as scope:
        opt = optimizer(lr)
        train_step = opt.minimize(loss)

    # Add evaluation (measure accuracy) to the graph
    with tf.name_scope("eval") as scope:
        if do_regression:
            acc = loss
        else:
            correct = tf.nn.in_top_k(z_out, Y, 1)
            acc = tf.reduce_mean(tf.cast(correct, tf.float32))
        tf.summary.scalar("accuracy", acc)

    summary = tf.summary.merge_all()

    # Execution
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)

        summary_writer = tf.summary.FileWriter("log/%s" % datetime.now().isoformat(), sess.graph)

        num_updates = N // mb
        for i in range(0, epochs):
            # Shuffle the inputs/outputs identically to construct a random batch
            rand_indices = np.random.permutation(np.arange(len(train['features'])))
            fullbatch_features = np.take(train['features'], rand_indices, axis=0)
            fullbatch_targets = np.take(train['targets'], rand_indices, axis=0)

            # Iterate over minibatches - ignore any leftover inputs
            for j in range(0, num_updates):
                minibatch_features = fullbatch_features[j*mb:(j+1)*mb]
                minibatch_targets = fullbatch_targets[j*mb:(j+1)*mb]
                sess.run(train_step, feed_dict={X: minibatch_features, Y: minibatch_targets})

                if verbose:
                    update_num = i * num_updates + j + 1
                    train_acc = sess.run(acc, feed_dict={X: train['features'], Y: train['targets']})
                    dev_acc = sess.run(acc, feed_dict={X: dev['features'], Y: dev['targets']})
                    print("Update %06d: train=%1.3f dev=%1.3f"%(update_num,train_acc,dev_acc), file=stderr) 

            # Calculate and report train and dev accuracy
            train_acc = sess.run(acc, feed_dict={X: train['features'], Y: train['targets']})
            dev_acc = sess.run(acc, feed_dict={X: dev['features'], Y: dev['targets']})
            print("Epoch %03d: train=%1.3f dev=%1.3f"%(i+1,train_acc,dev_acc), file=stderr) 

            # Add summaries to Tensorboard
            summary_str = sess.run(summary, feed_dict={X: train['features'], Y: train['targets']})
            summary_writer.add_summary(summary_str, i)
            summary_writer.flush()


if __name__ == '__main__':
    main()
