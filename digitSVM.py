# this script explores how well SVM does 

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager
import matplotlib.cm as cm
from sklearn import svm
from sklearn import cross_validation
import csv
import utils

# SVM takes very long to compute
# play with different kernels
# make prediction

# guaranteed that optimal C is also optimal for different set size?
# behavior of Learning Curve?
# why does SVD produce different eigenvalues each time?

def TrainSVM(X_train, X_val, X_test, y_train, y_val, y_test):

    '''
    This function finds the optimal parameters for the classifier
                                                                 
    Parameters:
    ----------
    X_train: numpy array (training set)
    X_val: numpy array (cross validation set)
    X_test: numpy array (test set)
    y_train: numpy array (class labels for training set)
    y_val: numpy array (class labels for validation set)
    y_test: numpy array (class lables for test set)

    Returns:
    --------
    optim: optimal parameters for the classifier
    '''


    # choose optimal penalty parameter
    cVal = [0.1,0.3,1.0,3.0,10.0,30.0,100.0]
    val_Scores = np.zeros(len(cVal))
    test_Scores = np.zeros(len(cVal))

    for i in xrange(0,len(cVal)):

        print '     Training with penalty parameter:', cVal[i]
        clf = svm.SVC(kernel='rbf', C=cVal[i])
        clf.fit(X_train[:1000,:],y_train[:1000]) # on a smaller subset because of time constraints
        val_Scores[i] = clf.score(X_val,y_val)
        test_Scores[i] = clf.score(X_test,y_test)
    
    print 
    optim = cVal[np.argmax(val_Scores)]
    print 'Optimal parameter is:',optim,'\n'
    print 'Test set score (on a smaller training set):', test_Scores[np.argmax(val_Scores)],'\n'

    # now plot learning curve to see whether more training data will help

    print 'Generating Learning Curve...'

    szV = range(50,10000,1000)
    LCvals=np.zeros((len(szV),3),dtype=np.float64) # store data points of learning curve
    clf = svm.SVC(kernel='rbf', C=optim)
    for i in xrange(0,len(szV)):
        print '     Training with data set size:', szV[i]
        clf.fit(X_train[:szV[i],:],y_train[:szV[i]])
        LCvals[i,0]=szV[i]
        LCvals[i,1]=clf.score(X_test,y_test)
        LCvals[i,2]=clf.score(X_train[:szV[i],:], y_train[:szV[i]])

    # generate figure (it shows that we're not in the high bias regime)
    fig = plt.figure(1, figsize=(10,10))
    prop = matplotlib.font_manager.FontProperties(size=15.5)
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(LCvals[:,0],1.0-LCvals[:,1],label='Test Set')
    ax.plot(LCvals[:,0],1.0-LCvals[:,2],label='Training Set')
    ax.set_ylabel(r"Error",fontsize=20)
    ax.set_xlabel(r"Training Set Size",fontsize=20)
    ax.axis([40.0, 10000, -0.1, 0.5])
    plt.legend(loc='upper right',prop=prop)
    plt.savefig('LC_SVM.pdf', bbox_inches='tight')
    fig.clear()

    return optim

def MakePred(X_train, y_train, optim, test, X_test, y_test):

    '''
    This function makes a prediction for the test set

    Parameters:
    -----------
    X_train: numpy array (training set)
    y_train: numpy array (class labels of training set)
    optim: optimal parameter for classification algorithm
    test: numpy array (test set)
    X_test: numpy array (set used to gauge performance - subset of initial training set)
    y_test: numpy array (class labels for X_test)

    '''

    # train model using optimal regularization
    print '     Training with 15000 samples...'
    clf = svm.SVC(kernel='rbf', C=optim)
    clf.fit(X_train[:15000,:],y_train[:15000]) # takes too long with more data points
    pred = clf.predict(test)
    print '     Score on test set:',clf.score(X_test,y_test)

    #print pred

    return pred


def main():
    
    # first read in data. tar == class label. dat == data
    print
    print "Reading Data..."
    dat, tar = utils.ReadDat("train.csv",1)
    print "Done Reading.\n"

    print 'Generated Sample Image\n'
    utils.sampleImage(dat[11,:])

    # preliminary statistics
    print "How many of each digit in training set?"
    print np.bincount(np.int32(tar)),'\n'

    # break up training data into training, cross validation, and test set (60,20,20)
    X_train, X_tmp, y_train, y_tmp = cross_validation.train_test_split(dat, tar, test_size=0.4)
    X_val, X_test, y_val, y_test = cross_validation.train_test_split(X_tmp, y_tmp, test_size=0.5)

    # now perform preprocessing (mean normalization and SVD to reduce feature space)
    print 'Preprocessing...'
    X_train = utils.meanNorm(X_train)
    X_val = utils.meanNorm(X_val)
    X_test = utils.meanNorm(X_test)
    U_r = utils.SVD(X_train,0.9)

    print np.shape(X_train), np.shape(X_val), np.shape(X_test)
    X_train = utils.Project(X_train,U_r)
    X_val = utils.Project(X_val,U_r)
    X_test = utils.Project(X_test,U_r)
    print np.shape(X_train), np.shape(X_val), np.shape(X_test)

    print 'Done Preprocessing.\n'

    # now train SVM on training set; choose optimal SVM parameters based on validation set
    print 'Now training SVM...\n'
    opt = TrainSVM(X_train, X_val, X_test, y_train, y_val, y_test)
    print 

    # now read in test set
    print "Reading Test Set..."
    test, tmp = utils.ReadDat("test.csv",0)
    print "Done Reading.\n"

    # Preprocess test set
    test = utils.meanNorm(test)
    test = utils.Project(test,U_r)

    # now make prediction
    print 'Making Prediction'
    pred = MakePred(X_train, y_train, opt, test, X_test, y_test)

    print
    print 'Outputting prediction\n'
    utils.OutputPred(pred,['ImageId','Label'],0)

    print 'Done'

if __name__ == '__main__':
    import sys
    main()

