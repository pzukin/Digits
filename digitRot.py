'''
This script explores how well we can distinguish digits
based on the ratio of a digits moment of inertia
eigenvalues.

'''

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager
import matplotlib.cm as cm
import csv
import utils
from sklearn import svm
from sklearn import cross_validation

def plotHistRot(datRot, tar):

    '''
    Plots distributions of the rotation independent features

    Parameters:
    -----------
    dat: numpy array of all records
    tar: numpy array with class labels for all records

    '''

    # generate figure
    matplotlib.rc('xtick.major', size=7.5)
    matplotlib.rc('xtick.minor', size=7.5)
    matplotlib.rc('ytick.major', size=7.5)
    matplotlib.rc('ytick.minor', size=7.5)
    fig = plt.figure(1, figsize = (10, 10))
    lsv=['-','-','-','-','-','-','-','--','--','--']
    prop = matplotlib.font_manager.FontProperties(size = 9.5)
    fs = 8

    for j in xrange(0,10):

        # pick out all records in a particular class
        mask = tar == np.float64(j)

        # make histogram of ratio of eigenvalues
        ax = fig.add_subplot(3, 3, 1)
        hist, bin_edges = np.histogram(datRot[mask,-1], bins = 30, density = True)
        ax.plot( 0.5 * (bin_edges[1:] + bin_edges[:-1]),
                 hist, lw = 2.0, label = str(j), ls = lsv[j])
        ax.axis([0, 40, 0, 0.8])
        ax.semilogx()
        ax.set_xlabel(r"Ratio of Rotation Eigenvalues", fontsize = fs)
        plt.legend(loc='upper right',prop = prop)

        # zeroth moment
        ax = fig.add_subplot(3, 3, 2)
        hist, bin_edges = np.histogram(datRot[mask,0], bins = 30, density = True)
        ax.plot( 0.5 * (bin_edges[1:] + bin_edges[:-1]),
                 hist, lw = 2.0, label = str(j), ls = lsv[j])
        ax.set_xlabel(r"0th moment (first radial bin)", fontsize = fs)

        # sin(theta) convolution
        ax = fig.add_subplot(3, 3, 3)
        hist, bin_edges = np.histogram(datRot[mask,1], bins = 30, density = True)
        ax.plot( 0.5 * (bin_edges[1:] + bin_edges[:-1]),
                 hist, lw = 2.0, label = str(j), ls = lsv[j])
        ax.set_xlabel(r"Sin $\theta$ Convolution (first radial bin)", fontsize = fs)

        # cos(theta) convolution                             
        ax = fig.add_subplot(3, 3, 4)
        hist, bin_edges = np.histogram(datRot[mask,2], bins = 30, density = True)
        ax.plot( 0.5 * (bin_edges[1:] + bin_edges[:-1]),
                 hist, lw = 2.0, label = str(j), ls = lsv[j])
        ax.set_xlabel(r"Cos $\theta$ Convolution (first radial bin)", fontsize = fs)

        # sin(2 * theta) convolution                             
        ax = fig.add_subplot(3, 3, 5)
        hist, bin_edges = np.histogram(datRot[mask,3], bins = 30, density = True)
        ax.plot( 0.5 * (bin_edges[1:] + bin_edges[:-1]),
                 hist, lw = 2.0, label = str(j), ls = lsv[j])
        ax.set_xlabel(r"Sin $2\theta$ Convolution (first radial bin)", fontsize = fs)

        # cos(2 * theta) convolution                             
        ax = fig.add_subplot(3, 3, 6)
        hist, bin_edges = np.histogram(datRot[mask,4], bins = 30, density = True)
        ax.plot( 0.5 * (bin_edges[1:] + bin_edges[:-1]),
                 hist, lw = 2.0, label = str(j), ls = lsv[j])
        ax.set_xlabel(r"Cos $2\theta$ Convolution (first radial bin)", fontsize = fs)


    plt.savefig('RotFeatures.pdf', bbox_inches = 'tight')
    fig.clear()

def TrainSVM(xTrain, xVal, xTest, yTrain, yVal, yTest):

    '''
    This function finds the optimal parameters for the classifier
                                                                 
    Parameters:
    ----------
    xTrain: numpy array (training set)    xVal: numpy array (cross validation set)
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
        clf.fit(xTrain[:1000,:], yTrain[:1000]) 
        valScores[i] = clf.score(xVal, yVal)
        testScores[i] = clf.score(xTest, yTest)
    
    print 
    optim = cVal[np.argmax(valScores)]
    print 'Optimal parameter is:', optim, '\n'
    print ('Test set score (on a smaller training set):', 
           testScores[np.argmax(valScores)])
    print

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

    print
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
    plt.savefig('LC_Rot_SVM.pdf', bbox_inches = 'tight')
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

    return pred


def main():

    # first read in data. tar == class label. dat == data
    print
    print "Reading Data..."
    dat, tar = utils.ReadDat("train.csv", 1)
    print "Done Reading.\n"

    print 'Calculating Rotation Indpendent Features...'
    datRot = np.zeros( (len(dat), 16), dtype = np.float64)
    for j in xrange(0,10):

        print '    Digit', j

        # pick out all records in a particular class 
        # doing it by class because running on whole
        # data set takes too long
        mask = tar == np.float64(j)
        datRot[mask, :] = utils.CalcRotFeatures(dat[mask])

    print datRot[0,:]
    print "Plot Distributions of Features...\n"
    plotHistRot(datRot, tar)

    print "Now training model on features to see performance... \n"
    # first mean normalization
    datRot = utils.meanNorm(datRot)
    # now split up training set into training, cross validation and test set
    xTrain, xTmp, yTrain, yTmp = cross_validation.train_test_split(datRot, tar, 
                                                                   test_size = 0.4)
    xVal, xTest, yVal, yTest = cross_validation.train_test_split(xTmp, yTmp, 
                                                                 test_size = 0.5)
    opt = TrainSVM(xTrain, xVal, xTest, yTrain, yVal, yTest)

    # now read in test set
    print "Reading Test Set..."
    test, tmp = utils.ReadDat("test.csv", 0)
    print "Done Reading.\n"

    # Preprocess test set
    print 'Calculating Rotation Independent Features for Test Set...'
    dl = len(test)/5
    testRot = np.zeros( (len(test), 16), dtype = np.float64)
    for i in xrange(0,5):
        print '     Chunk', i
        testRot[i * dl: (i + 1) * dl, :] = utils.CalcRotFeatures(test[i * dl: (i+ 1) * dl, :])

    print testRot[-1,:]
    testRot = utils.meanNorm(testRot)

    # now make prediction
    print 'Making Prediction'
    pred = MakePred(xTrain, yTrain, opt, testRot, xTest, yTest)

    print
    print 'Outputting prediction\n'
    utils.OutputPred(pred, ['ImageId','Label'], 2)


    print 'Done'


if __name__ == '__main__':
    main()
