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

def TrainSVM(xTrain, xVal, xTest, yTrain, yVal, yTest):

    '''
    This function finds the optimal parameters for the classifier
                                                                 
    Parameters:
    ----------
    xTrain: numpy array (training set)
    xVal: numpy array (cross validation set)
    xTest: numpy array (test set)
    yTrain: numpy array (class labels for training set)
    yVal: numpy array (class labels for validation set)
    yTest: numpy array (class lables for test set)

    Returns:
    --------
    optim: optimal parameters for the classifier
    '''


    # choose optimal penalty parameter
    cVal = [0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0]
    valScores = np.zeros(len(cVal))
    testScores = np.zeros(len(cVal))

    for i in xrange(0, len(cVal)):
	
        print '     Training with penalty parameter:', cVal[i]
        clf = svm.SVC(kernel = 'rbf', C = cVal[i])
        clf.fit(xTrain[:1000,:], yTrain[:1000]) # on a smaller subset because of time constraints
        valScores[i] = clf.score(xVal, yVal)
        testScores[i] = clf.score(xTest, yTest)
    
    print 
    optim = cVal[np.argmax(valScores)]
    print 'Optimal parameter is:', optim, '\n'
    print 'Test set score (on a smaller training set):', testScores[np.argmax(valScores)], '\n'

    # now plot learning curve to see whether more training data will help

    print 'Generating Learning Curve...'

    szV = range(50, 10000, 1000)
    LCvals = np.zeros((len(szV), 3),dtype = np.float64) # store data points of learning curve
    clf = svm.SVC(kernel = 'rbf', C = optim)
    for i in xrange(0, len(szV)):
        print '     Training with data set size:', szV[i]
        clf.fit(xTrain[:szV[i],:], yTrain[:szV[i]])
        LCvals[i,0] = szV[i]
        LCvals[i,1] = clf.score(xTest, yTest)
        LCvals[i,2] = clf.score(xTrain[:szV[i],:], yTrain[:szV[i]])

    # generate figure (it shows that we're not in the high bias regime)
    fig = plt.figure(1, figsize = (10, 10))
    prop = matplotlib.font_manager.FontProperties(size = 15.5)
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(LCvals[:,0], 1.0 - LCvals[:,1], label = 'Test Set')
    ax.plot(LCvals[:,0], 1.0 - LCvals[:,2], label = 'Training Set')
    ax.set_ylabel(r"Error", fontsize = 20)
    ax.set_xlabel(r"Training Set Size", fontsize = 20)
    ax.axis([40.0, 10000, -0.1, 0.5])
    plt.legend(loc = 'upper right', prop = prop)
    plt.savefig('LC_SVM.pdf', bbox_inches = 'tight')
    fig.clear()

    return optim

def MakePred(xTrain, yTrain, optim, test, xTest, yTest):

    '''
    This function makes a prediction for the test set

    Parameters:
    -----------
    xTrain: numpy array (training set)
    yTrain: numpy array (class labels of training set)
    optim: optimal parameter for classification algorithm
    test: numpy array (test set)
    xTest: numpy array (set used to gauge performance - subset of initial training set)
    yTest: numpy array (class labels for xTest)

    '''

    # train model using optimal regularization
    print '     Training with 15000 samples...'
    clf = svm.SVC(kernel = 'rbf', C = optim)
    clf.fit(xTrain[:15000,:], yTrain[:15000]) # takes too long with more data points
    pred = clf.predict(test)
    print '     Score on test set:', clf.score(xTest, yTest)

    #print pred

    return pred


def main():
    
    # first read in data. tar == class label. dat == data
    print
    print "Reading Data..."
    dat, tar = utils.ReadDat("train.csv", 1)
    print "Done Reading.\n"

    print 'Generated Sample Image\n'
    utils.sampleImage(dat[11, :])

    # preliminary statistics
    print "How many of each digit in training set?"
    print np.bincount(np.int32(tar)), '\n'

    # break up training data into training, cross validation, and test set (60,20,20)
    xTrain, xTmp, yTrain, yTmp = cross_validation.train_test_split(dat, tar, 
                                                                   test_size = 0.4)
    xVal, xTest, yVal, yTest = cross_validation.train_test_split(xTmp, yTmp, 
                                                                 test_size = 0.5)

    # now perform preprocessing (mean normalization and SVD to reduce feature space)
    print 'Preprocessing...'
    xTrain = utils.meanNorm(xTrain)
    xVal = utils.meanNorm(xVal)
    xTest = utils.meanNorm(xTest)
    Ur = utils.SVD(xTrain, 0.9)

    print np.shape(xTrain), np.shape(xVal), np.shape(xTest)
    xTrain = utils.Project(xTrain, Ur)
    xVal = utils.Project(xVal, Ur)
    xTest = utils.Project(xTest, Ur)
    print np.shape(xTrain), np.shape(xVal), np.shape(xTest)

    print 'Done Preprocessing.\n'

    # now train SVM on training set; choose optimal SVM parameters based on validation set
    print 'Now training SVM...\n'
    opt = TrainSVM(xTrain, xVal, xTest, yTrain, yVal, yTest)
    print 

    # now read in test set
    print "Reading Test Set..."
    test, tmp = utils.ReadDat("test.csv", 0)
    print "Done Reading.\n"

    # Preprocess test set
    test = utils.meanNorm(test)
    test = utils.Project(test, Ur)

    # now make prediction
    print 'Making Prediction'
    pred = MakePred(xTrain, yTrain, opt, test, xTest, yTest)

    print
    print 'Outputting prediction\n'
    utils.OutputPred(pred, ['ImageId','Label'], 0)

    print 'Done'

if __name__ == '__main__':
    import sys
    main()

