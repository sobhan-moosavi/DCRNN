import numpy as np
import random
import math
from scipy import stats
import time
import cPickle
import time

from sklearn.preprocessing import OneHotEncoder
import functools

from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--shape', type=int, nargs='+', default=[50, 50])
parser.add_argument('--thresh', type=float, default=0.2)
parser.add_argument('--test', type=str, default='small')
parser.add_argument('--version', type=str, default='v1')
parser.add_argument('--suffix', type=str, default='SpdAclRp')

args = parser.parse_args()
shape = args.shape
thresh = args.thresh
test_type = args.test
version = args.version

def load_data(file):
    trip_segments = np.load(file)/10.0
    print("Number of samples: {}".format(trip_segments.shape[0]))
    return trip_segments


# In[4]:


def returnTrainDevTestData():
    
    vers = ''
    if version == 'v2': vers = '_v2'        
    if test_type=='small':
        file_name = 'data2/dissimilar_trajectories_{}_{}_{}_ndl_10{}'.format(thresh, shape[0], shape[1], vers)
    elif test_type=='large': 
        file_name = 'data3/dissimilar_trajectories_{}_{}_ndl_10{}'.format(shape[0], shape[1], vers)
    elif test_type=='random': 
        file_name = 'data4/random_trajectories_{}_{}_ndl_10{}'.format(shape[0], shape[1], vers)
    elif test_type=='test_rand': 
        file_name = 'data5/th_train_rnd_test_{}_{}_{}_ndl_10{}'.format(thresh, shape[0], shape[1], vers)
    matrices = load_data(file_name + '.npy')
    keys = cPickle.load(open(file_name + '.pkl'))     

    #load trajectory to set mapping
    f_name = 'data2/driverToTrajectoryToSet_{}_{}_{}_{}.pkl'.format(thresh, shape[0], shape[1], args.suffix)  #this is fine with "th_train_rnd_test" type!
    if test_type == 'large':
        f_name = 'data3/driverToTrajectoryToSet_{}_{}_{}.pkl'.format(shape[0], shape[1], args.suffix)
    elif test_type == 'random': 
        f_name = 'data4/driverToTrajectoryToSet_{}_{}_{}.pkl'.format(shape[0], shape[1], args.suffix)
    driverToTrajectoryToSet = cPickle.load(open(f_name ))
    
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
        assign = 'None'
        if t in driverToTrajectoryToSet[d]: assign = (driverToTrajectoryToSet[d])[t]
        if t != curTraj:
            curTraj = t
            r = random.random()  
        #if r < .8:        
        if assign == 'train':
            train_data.append(m)
            train_labels.append(dr)
        elif args.test == 'test_rand' and assign == 'None':    
            test_data.append(m)
            test_labels.append(dr)        
        elif args.test != 'test_rand' and assign == 'test': 
            test_data.append(m)
            test_labels.append(dr)
    
    print("number of drivers "+str(len(driverIds)))
    train_data   = np.asarray(train_data, dtype="float32")
    train_labels = np.asarray(train_labels, dtype="int32")
    test_data    = np.asarray(test_data, dtype="float32")
    test_labels  = np.asarray(test_labels, dtype="int32")
    
    train_data, train_labels = shuffle_in_union(train_data, train_labels)
  
    return train_data, train_labels, test_data, test_labels, len(driverIds)


# In[5]:


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
    from sklearn.ensemble import GradientBoostingClassifier
    if test_type == 'small' or shape[0]==50:
        parameters_gbdt = {'learning_rate':[0.005, 0.01, 0.05, 0.06, 0.07], 'n_estimators': [250, 300, 350, 400, 500, 600], 'max_depth' :[2, 3,4,5, 6]}
    else:
        parameters_gbdt = {'learning_rate':[0.005, 0.01, 0.05], 'n_estimators': [400, 500, 600], 'max_depth' :[2,3,4,5]}
        #parameters_gbdt = {'learning_rate':[0.005, 0.01, 0.05, 0.06], 'n_estimators': [300, 400, 500, 600], 'max_depth' :[2,3,4,5,6]}
        
    gbdt = GridSearchCV(estimator = GradientBoostingClassifier( max_features='sqrt', subsample=0.8, random_state=10), cv=5, param_grid = parameters_gbdt, verbose=15, scoring='accuracy',n_jobs=25)
    gbdt.fit(train, train_labels)
    
    print('\n\n ******* Best Parameter Set ********* \n')
    print (gbdt.best_params_)
    print ('\n\n')
    
    predictions_gbdt = gbdt.predict(test)
    #print(predictions_gbdt)

    print("############# GBDT #############")
    count1 = 0
    for i in range(0,len(predictions_gbdt)):
        if (test_labels[i] == predictions_gbdt[i]):
             count1 = count1 +1
        #else: print(test_labels[i], predictions_gbdt[i])
    print(count1)
    print ('Final Test Accuracy: {:.2f}%'.format(100.0*count1/float(len(test))))
    #print(gbdt.cv_results_)
    #print(gbdt.best_estimator_)
