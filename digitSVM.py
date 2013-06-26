# this script explores how well SVM does 

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager
import matplotlib.cm as cm
from sklearn import svm
from sklearn import cross_validation
import csv

# SVM takes very long to compute
# play with different kernels
# make prediction

# guaranteed that optimal C is also optimal for different set size?
# behavior of Learning Curve?
# why does SVD produce different eigenvalues each time?

def ReadDat(path,ch):

    # using genfromtxt takes too long so I'm doing it differently

    f = open(path)
    file = f.read()
    f.close()
    
    # neglecting first labeled row
    lines = file.splitlines()[1:]
    
    # will save data in new data structure
    # know dimensions beforehand                                                                                           
    # row labels different samples, column labels different pixels                                                         
    dat = np.zeros((len(lines),784),dtype = np.float64)
    tar = np.zeros(len(lines),dtype = np.float64)

    for i in xrange(0,len(lines)):
        tmp = np.float64(lines[i].split(","))
        dat[i,:] = tmp[ch:]
        tar[i] = tmp[0] # only relevan when ch=1 (training set)

    return dat, tar

def PreProcess(dat):

    # first mean normalization
    print 'Feature normalization'
    dat = meanNorm(dat)

    # now SVD (ML training took too long to run on whole data set)
    print 'SVD'
    cov = np.cov(dat.T,bias=1) # covariance matrix
    U, s, V = np.linalg.svd(cov, full_matrices=True)

    # now choose dimension of projection so that we retain 90% of variance
    for i in xrange(1,len(s)+1):
        if (np.sum(s[:i])/np.sum(s) > 0.90):
            break

    print i, 'features. Used to be', np.shape(dat)[1]
    U_r = U[:,:i] # reduced matrix U that projects onto smaller dimensional space
    dat = np.dot(dat,U_r)

    return dat, U_r

def meanNorm(dat):
# mean Normalization

    mu = np.mean(dat, axis=0)
    sigma = np.std(dat, axis=0)
    dat = (dat-mu)/sigma

    # some feautures have mu=0.0, sigma=0.0. force all those features to be zero                         
    dat[np.isnan(dat)]=0.0

    return dat

def TrainSVM(X_train, X_val, X_test, y_train, y_val, y_test, U_r):

    # first perform mean normalization on validation and test set
    X_val = meanNorm(X_val)
    X_test = meanNorm(X_test)

    # now project onto smaller dimensional space
    X_val = np.dot(X_val,U_r)
    X_test = np.dot(X_test,U_r)

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
    plt.savefig('LC.pdf', bbox_inches='tight')
    fig.clear()

    return optim

def samIm(vals):

    fig = plt.figure(1, figsize=(10,10))
    ax = fig.add_subplot(1, 1, 1)
    ax.imshow(np.reshape(vals,(28,28)), cmap = cm.Greys_r)
    plt.savefig('sample.pdf', bbox_inches='tight')
    fig.clear()

def MakePred(X_train, y_train, optim, test, U_r, X_test, y_test):

    test = meanNorm(test) # mean normalization
    test = np.dot(test,U_r) # project into smaller subspace
    
    # using part of training set to see possible behavior
    X_test = meanNorm(X_test)
    X_test = np.dot(X_test,U_r)

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
    dat, tar = ReadDat("train.csv",1)
    print "Done Reading.\n"

    print 'Generated Sample Image\n'
    samIm(dat[11,:])

    # preliminary statistics
    print "How many of each digit in training set?"
    print np.bincount(np.int32(tar)),'\n'

    # break up training data into training, cross validation, and test set (60,20,20)
    X_train, X_tmp, y_train, y_tmp = cross_validation.train_test_split(dat, tar, test_size=0.4)
    X_val, X_test, y_val, y_test = cross_validation.train_test_split(X_tmp, y_tmp, test_size=0.5)

    # now perform preprocessing (mean normalization and SVD to reduce feature space)
    print 'Preprocessing...'
    X_train, U_r = PreProcess(X_train)
    print 'Done Preprocessing.\n'

    # now train SVM on training set; choose optimal SVM parameters based on validation set
    print 'Now training SVM...\n'
    #print np.shape(X_train)
    opt = TrainSVM(X_train, X_val, X_test, y_train, y_val, y_test, U_r)
    print 

    # now read in test set
    print "Reading Test Set..."
    test, tmp = ReadDat("test.csv",0)
    print "Done Reading.\n"

    # now make prediction
    print 'Making Prediction'
    pred = MakePred(X_train, y_train, opt, test, U_r, X_test, y_test)

    print
    print 'Outputting prediction\n'
    open_file_object = csv.writer(open("predSVM.csv", "wb"))
    for p in pred:
        open_file_object.writerow(str(int(p)))

    print 'Done'

if __name__ == '__main__':
    import sys
    main()

