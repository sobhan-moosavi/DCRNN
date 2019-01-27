'''
    This code is to use Gradient Boosting Decision Tree model for driver identification/prediction. 
    The input for this model is either a vector of length 321, called V1 (as described by Dong et al. 2016), or the modified version which is of size 384, called V2 (as described in our D-CRNN paper). 
    As input, you need to specify type of feature vector as input. 
'''

import functools
import numpy as np
import random
import cPickle
from sklearn.ensemble import GradientBoostingClassifier
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--version', type=str, default='v1')  # use v1 for original feature vector (size:321), and v2 for the modified version (size:384)

args = parser.parse_args()
version = args.version

# load pre-constructed input data files. 
def load_data(file):
    trip_segments = np.load(file)
    print("Number of samples: {}".format(trip_segments.shape[0]))
    return trip_segments


# to prepare train and test sets; 85% of data will be used for train and 15% for test. 
def returnTrainDevTestData():
        
    matrices = load_data('data/non_deep_features_{}.npy'.format(version))
    keys = cPickle.load(open('data/non_deep_features_{}.pkl'.format(version), 'rb'))
    
    #Build Train, Dev, Test sets
    train_data = []
    train_labels = []
    test_data = []
    test_labels = []
    
    curTraj = ''
    r = 0
    
    driverIds = {}
    
    for idx in range(len(keys)):
        d,t = keys[idx]
        if d in driverIds:
            dr = driverIds[d]
        else: 
            dr = len(driverIds)
            driverIds[d] = dr
        m = matrices[idx]  
        if t != curTraj:
            curTraj = t
            r = random.random()  
        if r < 0.85: 
            train_data.append(m)
            train_labels.append(dr)
        else: 
            test_data.append(m)
            test_labels.append(dr)
    
    print("number of drivers "+str(len(driverIds)))
    train_data   = np.asarray(train_data, dtype="float32")
    train_labels = np.asarray(train_labels, dtype="int32")
    test_data    = np.asarray(test_data, dtype="float32")
    test_labels  = np.asarray(test_labels, dtype="int32")
    
    train_data, train_labels = shuffle_in_union(train_data, train_labels)
  
    return train_data, train_labels, test_data, test_labels, len(driverIds)


# to simultaneously shuffle two sets
def shuffle_in_union(a, b):
    assert len(a) == len(b)
    shuffled_a = np.empty(a.shape, dtype=a.dtype)
    shuffled_b = np.empty(b.shape, dtype=b.dtype)
    permutation = np.random.permutation(len(a))
    for old_index, new_index in enumerate(permutation):
        shuffled_a[new_index] = a[old_index]
        shuffled_b[new_index] = b[old_index]
    return shuffled_a, shuffled_b



if __name__ == '__main__':
    train, train_labels, test, test_labels, num_classes = returnTrainDevTestData()
    print("number of train_data "+str(train_labels.shape))
    print("number of test_data "+str(test_labels.shape))
    
    from sklearn.model_selection import GridSearchCV
    
    
    ############################ BOOST DECISION TREE #####################    
    # specifying the grid search space for three parameters of GBDT model
    parameters_gbdt = {'learning_rate':[0.005, 0.01, 0.05, 0.06, 0.07], 'n_estimators': [250, 300, 350, 400, 500, 600], 'max_depth' :[2, 3,4,5, 6]}
        
    gbdt = GridSearchCV(estimator = GradientBoostingClassifier( max_features='sqrt', subsample=0.8, random_state=10), cv=5, param_grid = parameters_gbdt, verbose=15, scoring='accuracy',n_jobs=5)
    gbdt.fit(train, train_labels)
    
    print('\n\n ******* Best Parameter Set ********* \n')
    print (gbdt.best_params_)
    print ('\n\n')
    
    predictions_gbdt = gbdt.predict(test)

    print("############# GBDT #############")
    count1 = 0
    for i in range(0,len(predictions_gbdt)):
        if (test_labels[i] == predictions_gbdt[i]):
             count1 = count1 +1
    
    print ('Final Test Accuracy: {:.2f}%'.format(100.0*count1/float(len(test))))    
