"""
This is an implementation of CNN architecture which is presented in "characterizing driving styles with deep learning", Dong et al. (2016). 
Author: Sobhan Moosavi
"""

from __future__ import print_function
from __future__ import division
import tensorflow as tf
from tensorflow.contrib import rnn

import numpy as np
import random
import math
from scipy import stats
import time
import cPickle
import time

from sklearn.preprocessing import OneHotEncoder
import functools
import argparse
import os
import shutil

# some parameters as input
parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=125)
parser.add_argument('--neurons', type=int, default=100)
args = parser.parse_args()
epochs    = args.epochs


# helper class
def lazy_property(function):
    attribute = '_' + function.__name__

    @property
    @functools.wraps(function)
    def wrapper(self):
        if not hasattr(self, attribute):
            setattr(self, attribute, function(self))
        return getattr(self, attribute)
    return wrapper


# setting up the architecture
class CNN_MODEL:

    def __init__(self, data, target, dropout):
        self.data = data
        self.target = target
        self._dropout = dropout
        self.cost
        self.prediction
        self.optimize
        self.accuracy
        self.predProbs

    @lazy_property
    def prediction(self):    
        # Input Layer
        # Reshape X to 4-D tensor: [batch_size, width, height, channels]
        # Trajectory Segments are FEATURESx128, and we just have one channel. 
        input_layer = tf.reshape(self.data, [-1, FEATURES, 128, 1])

        # Convolutional Layer #1
        # Computes 32 features using a FEATURESx5 filter with Sigmoid activation. [convolution is over time]
        # Padding is added to preserve width and height.
        # Input Tensor Shape: [batch_size, FEATURES, 128, 1]
        # Output Tensor Shape: [batch_size, 1 124, 32]
        conv1 = tf.layers.conv2d(inputs=input_layer, filters=32, kernel_size=[FEATURES, 5], strides=1, activation=tf.nn.sigmoid)

        # Pooling Layer #1
        # First max pooling layer with a 1x2 filter and stride of 2
        # Input Tensor Shape: [batch_size, 1, 124, 32]
        # Output Tensor Shape: [batch_size, 1, 62, 32]
        pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[1, 2], strides=2)

        # Convolutional Layer #2
        # Computes 64 features using a 1x3 filter.
        # Padding is added to preserve width and height.
        # Input Tensor Shape: [batch_size, 1, 62, 32]
        # Output Tensor Shape: [batch_size, 1, 60, 64]
        conv2 = tf.layers.conv2d(inputs=pool1, filters=64, kernel_size=[1, 3], strides=1, activation=tf.nn.sigmoid)

        # Pooling Layer #2
        # Second max pooling layer with a 1x2 filter and stride of 2
        # Input Tensor Shape: [batch_size, 1, 60, 64]
        # Output Tensor Shape: [batch_size, 1, 30, 64]
        pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[1, 2], strides=2)

        # Convolutional Layer #3
        # Computes 64 features using a 1x3 filter.
        # Padding is added to preserve width and height.
        # Input Tensor Shape: [batch_size, 1, 30, 64]
        # Output Tensor Shape: [batch_size, 1, 28, 64]
        conv3 = tf.layers.conv2d(inputs=pool2, filters=64, kernel_size=[1, 3], strides=1, activation=tf.nn.sigmoid)

        # Pooling Layer #3
        # Third max pooling layer with a 1x2 filter and stride of 2
        # Input Tensor Shape: [batch_size, 1, 28, 64]
        # Output Tensor Shape: [batch_size, 1, 14, 64]
        pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[1, 2], strides=2)

        # Flatten tensor into a batch of vectors
        # Input Tensor Shape: [batch_size, 1, 14, 64]
        # Output Tensor Shape: [batch_size, 1 * 14 * 64]
        pool3_flat = tf.reshape(pool3, [-1, 1 * 14 * 64])

        # Dense Layer #1
        # Densely connected layer with 128 neurons
        # Input Tensor Shape: [batch_size, 7 * 7 * 64]
        # Output Tensor Shape: [batch_size, 128]
        dense1 = tf.layers.dense(inputs=pool3_flat, units=args.neurons, activation=tf.nn.sigmoid)

        # Dropout Layer #1
        # Add dropout operation; (1-rate) probability that element will be kept
        dropout1 = tf.layers.dropout(inputs=dense1, rate=self._dropout)

        # Dense Layer #2
        # Densely connected layer with 128 neurons
        # Input Tensor Shape: [batch_size, 7 * 7 * 64]
        # Output Tensor Shape: [batch_size, 128]
        dense2 = tf.layers.dense(inputs=dropout1, units=args.neurons, activation=tf.nn.sigmoid)

        # Dropout Layer #2
        # Add dropout operation; (1-rate) probability that element will be kept
        dropout2 = tf.layers.dropout(inputs=dense2, rate=self._dropout)

        # Logits layer
        # Input Tensor Shape: [batch_size, 128]
        # Output Tensor Shape: [batch_size, numOfDrivers]
        logits = tf.layers.dense(inputs=dropout2, units=int(self.target.get_shape()[1]), activation=None) 
                
        predicted_classes = tf.argmax(input=logits, axis=1)
        softmax_prob = tf.nn.softmax(logits, name="softmax_tensor")
        
        return logits, predicted_classes, softmax_prob
    
    @lazy_property
    def cost(self):
        logits, predicted_classes, softmax_prob = self.prediction               
        cross_entropy = tf.reduce_mean(-tf.reduce_sum(self.target * tf.log(softmax_prob), reduction_indices=[1]))
        return cross_entropy

    @lazy_property
    def optimize(self):       
        optimizer = tf.train.MomentumOptimizer(learning_rate=0.05, momentum=0.9, use_nesterov=True)
        return optimizer.minimize(self.cost)
    
    @lazy_property
    def accuracy(self):
        logits, predicted_classes, softmax_prob = self.prediction
        correct_pred = tf.equal(tf.argmax(self.target, 1), tf.argmax(softmax_prob, 1))
        return tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        
    @lazy_property
    def predProbs(self):
        logits, predicted_classes, softmax_prob = self.prediction
        return softmax_prob


# load pre-constructed feature matrices
def load_data(file):
    trip_segments = np.load(file)
    print("Number of samples: {}".format(trip_segments.shape[0]))
    return trip_segments
  
 
# to split data to train, dev, and test; default: 75% train, 10% dev, and 15% test
def returnTrainDevTestData():

    matrices = load_data('data/RandomSample_5_10.npy')
    keys = cPickle.load(open('data/RandomSample_5_10.pkl', 'rb'))
        
    FEATURES = matrices.shape[-1]    
    
    #Build Train, Dev, Test sets
    train_data = []
    train_labels = []
    dev_data = []
    dev_labels = []
    test_data = []
    test_labels = []
    test_tripId = []
    
    curTraj = ''
    assign = ''
    
    driverIds = {}
    
    for idx in range(len(keys)):
        d,t = keys[idx]
        if d in driverIds:
            dr = driverIds[d]
        else: 
            dr = len(driverIds)
            driverIds[d] = dr            
        m = matrices[idx][1:129,]
        if t != curTraj:
            curTraj = t
            r = random.random()
        m = np.transpose(m) #need this step and the next for CNN
        m = np.reshape(m, FEATURES*128)
        if r < 0.75:
            train_data.append(m)
            train_labels.append(dr)
        elif r < 0.85:
            dev_data.append(m)
            dev_labels.append(dr)
        else: 
            test_data.append(m)
            test_labels.append(dr)      
            test_tripId.append(t)

    train_data   = np.asarray(train_data, dtype="float32")
    train_labels = np.asarray(train_labels, dtype="int32")
    dev_data   = np.asarray(dev_data, dtype="float32")
    dev_labels = np.asarray(dev_labels, dtype="int32")
    test_data    = np.asarray(test_data, dtype="float32")
    test_labels  = np.asarray(test_labels, dtype="int32")
    
    rng_state = np.random.get_state()
    np.random.set_state(rng_state)
    np.random.shuffle(train_data)
    np.random.set_state(rng_state)
    np.random.shuffle(train_labels)
  
    return train_data, train_labels, dev_data, dev_labels, test_data, test_labels, test_tripId, len(driverIds), FEATURES


    
def convertLabelsToOneHotVector(labels, ln):
    tmp_lb = np.reshape(labels, [-1,1])
    next_batch_start = 0
    _x = np.arange(ln)
    _x = np.reshape(_x, [-1, 1])
    enc = OneHotEncoder()
    enc.fit(_x)
    labels =  enc.transform(tmp_lb).toarray()
    return labels

    
def returnTripLevelAccuracy(test_labels, test_tripId, probabilities, num_classes):    
    lbl = ''
    probs = []
    correct = total = 0
    for i in range(len(test_labels)):
        if lbl == test_tripId[i]:
            probs.append(probabilities[i])
        else:
            if len(probs) > 0:
                total += 1.0
                probs = np.asarray(probs)
                probs = np.mean(probs, axis=0)
                probs = (probs/np.max(probs)).astype(int)
                if np.sum(probs&test_labels[i-1].astype(int)) == 1: correct += 1
            probs = []
            probs.append(probabilities[i])
            lbl = test_tripId[i]
    if len(probs) > 0:
        total += 1.0
        probs = np.asarray(probs)
        probs = np.mean(probs, axis=0)
        probs = (probs/np.max(probs)).astype(int)        
        #print probs, test_labels[len(test_labels)-1]
        if np.sum(probs&test_labels[len(test_labels)-1].astype(int))==1: correct += 1
        
    return correct/total
 
 
if __name__ == '__main__':
    
    ITERATIONS = 3  #number of times to repeat the whole experiment 
    ALL_SEG_ACC = []
    ALL_TRP_ACC = []
    
    for IT in range(0, ITERATIONS):
        tf.reset_default_graph()
        print ('\n\n************ Iteration: {} ************\n'.format(IT+1))
        
        # shape = [500, 50]
        st = time.time()
        train, train_labels, dev, dev_labels, test, test_labels, test_tripId, num_classes, FEATURES = returnTrainDevTestData()
        
        display_step = 100
        training_steps = 1000000
        batch_size = 256
        
        train_dropout = 0.5
        test_dropout = 0.0
        
        timesteps = 128 # Number of rows in Matrix of a Segment
        
        train_labels = convertLabelsToOneHotVector(train_labels, num_classes)  
        dev_labels = convertLabelsToOneHotVector(dev_labels, num_classes)  
        test_labels  = convertLabelsToOneHotVector(test_labels, num_classes)    
        
        print('All data is loaded in {:.1f} seconds'.format(time.time()-st))
        print('There are {} example in train, {} in dev, and {} in test set'.format(len(train), len(dev), len(test)))
        print('num_classes', num_classes)
        
        data = tf.placeholder(tf.float32, [None, FEATURES*128], name='data')    
        target = tf.placeholder(tf.float32, [None, num_classes], name='target')
        dropout = tf.placeholder(tf.float32)
        model = CNN_MODEL(data, target, dropout)
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        
        train_start = time.time()
        start = time.time()
        next_batch_start = 0
        
        steps_to_epoch = len(train)/batch_size
        
        maxDevAccuracy = 0.0 #This will be used as a constraint to save the best model
        minDevLoss = 1000.0
        bestEpoch = 0
        
        saver = tf.train.Saver() #This is the saver of the model    
        model_name = 'models/CRNN_model/'        
        if os.path.exists(model_name):
            shutil.rmtree(model_name)            
        os.makedirs(model_name)
        
        for step in range(training_steps):
            idx_end = min(len(train),next_batch_start+batch_size)        
            sess.run(model.optimize, {data: train[next_batch_start:idx_end,:], target: train_labels[next_batch_start:idx_end,:], dropout: train_dropout})                
            
            epoch = int(step/steps_to_epoch)
            if epoch > epochs: break  #epochs: maximum possible epochs 
            
            if epoch > bestEpoch or epoch == 0:
                acc = sess.run(model.accuracy, {data: dev, target: dev_labels, dropout: test_dropout})
                if epoch > 5 and acc > maxDevAccuracy:
                    d_loss  = sess.run(model.cost, {data: dev, target: dev_labels, dropout: test_dropout}) 
                    maxDevAccuracy = acc
                    minDevLoss = d_loss
                    bestEpoch = epoch
                    save_path = saver.save(sess, model_name)
                    print('Model saved in path: {}, Dev Loss: {:.2f}, Dev Accuracy: {:.2f}%, Epoch: {:d}'.format(save_path, minDevLoss, 100*maxDevAccuracy, epoch))
              
            if step % display_step == 0:            
                loss = sess.run(model.cost, {data: train[next_batch_start:idx_end,:], target: train_labels[next_batch_start:idx_end,:], dropout: test_dropout})
                train_acc = sess.run(model.accuracy, {data: train[next_batch_start:idx_end,:], target: train_labels[next_batch_start:idx_end,:], dropout: test_dropout})
                dev_acc = sess.run(model.accuracy, {data: dev, target: dev_labels, dropout: test_dropout})
                dev_loss  = sess.run(model.cost, {data: dev, target: dev_labels, dropout: test_dropout})              
                print('Step {:2d}, Epoch {:2d}, Train Loss {:.3f}, Dev-Loss {:.3f}, Mini-Batch Train_Accuracy {:.1f}%, Dev-Accuracy {:.1f}%, ({:.1f} sec)'.format(step + 1, epoch, loss, dev_loss, 100*train_acc, 100*dev_acc, (time.time()-start)))            
                start = time.time()
                
            next_batch_start = next_batch_start+batch_size        
            if next_batch_start >= len(train):
                rng_state = np.random.get_state()
                np.random.set_state(rng_state)
                np.random.shuffle(train)
                np.random.set_state(rng_state)
                np.random.shuffle(train_labels)
                next_batch_start = 0
        
        
        print("Optimization Finished!")
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, model_name)
        accuracy = sess.run(model.accuracy, {data: test, target: test_labels, dropout: test_dropout})
        # calculate trip-level prediction accuracy
        probabilities = sess.run(model.predProbs, {data: test, target: test_labels, dropout: test_dropout})
        trip_level_accuracy = returnTripLevelAccuracy(test_labels, test_tripId, probabilities, num_classes)
        print('Test-Accuracy(segment): {:.2f}%, Test-Accuracy(trip): {:.2f}%,Train-Time: {:.1f}sec'.format(accuracy*100, trip_level_accuracy*100, (time.time()-train_start)))
        print('Best Dev-Accuracy: {:.2f}%, Least Dev-Loss: {:.2f}, Best Epoch: {}'.format(maxDevAccuracy*100, minDevLoss, bestEpoch))

        ALL_SEG_ACC.append(accuracy*100)
        ALL_TRP_ACC.append(trip_level_accuracy*100)
        
    
    print ('\n\nAll Iterations are completed!')
    print ('Average Segment Accuracy: {:.2f}%, Std: {:.2f}, Min: {:.2f}, Max: {:.2f}'.format(np.mean(ALL_SEG_ACC), np.std(ALL_SEG_ACC), np.min(ALL_SEG_ACC), np.max(ALL_SEG_ACC)))
    print ('Average Trip Accuracy: {:.2f}%, Std: {:.2f}, Min: {:.2f}, Max: {:.2f}'.format(np.mean(ALL_TRP_ACC), np.std(ALL_TRP_ACC), np.min(ALL_TRP_ACC), np.max(ALL_TRP_ACC)))