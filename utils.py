# these functions are useful utilities that are used for different training algorihms.

import numpy as np
import csv
import matplotlib.pyplot as plt
import matplotlib.font_manager
import matplotlib.cm as cm

def ReadDat(path, ch):

    '''
    Reads in data from path and stores in 2 data structures

    Parameters:
    ----------
    path: specifies path of data file
    ch: specifies whether file is training or test set (ch == 1 or 0)

    Returns:
    --------
    dat: nunmpy array with all records
    tar: numpy array with class labels of all records (not relevant for test set case) 

    '''

    f = open(path)
    file = f.read()
    f.close()

    # neglecting first labeled row
    lines = file.splitlines()[1:]

    # will save data in new data structure
    # know dimensions beforehand
    # row labels different samples, column labels different pixels

    dat = np.zeros((len(lines), 784), dtype = np.float64)
    tar = np.zeros(len(lines), dtype = np.float64)

    for i in xrange(0, len(lines)):
        tmp = np.float64(lines[i].split(","))
        dat[i, :] = tmp[ch:]
        tar[i] = tmp[0] # only relevant when ch = 1 (training set)                         

    return dat, tar


def meanNorm(dat):

    '''
    This function subtracts the mean from each feature and normalizes by the std.
    
    Parameters:
    ----------
    dat: numpy array with all records
 
    Returns:
    --------
    dat: numpy array with all records mean normalized

    '''

    mu = np.mean(dat, axis = 0)
    sigma = np.std(dat, axis = 0)
    dat = (dat - mu) / sigma

    # some feautures have mu = 0.0, sigma = 0.0. force all those features to be zero 

    dat[np.isnan(dat)] = 0.0

    return dat


def sampleImage(vals):

    '''
    This function generates a sample image

    Parameters:
    ----------
    vals: numpy array of all the pixel values for a given record
    

    '''

    fig = plt.figure(1, figsize = (10, 10))
    ax = fig.add_subplot(1, 1, 1)
    ax.imshow(np.reshape(vals, (28, 28)), cmap = cm.Greys_r)
    plt.savefig('sample.pdf', bbox_inches = 'tight')
    fig.clear()


def SVD(dat, var):

    '''
    This function performs SVD on the data set and returns a projection vector that keeps a certain percentage (var) of the variance.

    Parameters:
    -----------
    dat: numpy array with all records
    var: specifies what percentage of variance to keep

    Returns:
    --------
    Ur: projection matrix

    '''

    cov = np.cov(dat.T, bias = 1) # covariance matrix  
    U, s, V = np.linalg.svd(cov, full_matrices = True)

    # now choose dimension of projection so that we retain var% of variance                      
    for i in xrange(1, len(s) + 1):
        if (np.sum(s[:i]) / np.sum(s) > var):
            break
        
    print i, 'features. Used to be', np.shape(dat)[1]
    Ur = U[:, :i] # reduced matrix U that projects onto smaller dimensional space

    return Ur


def Project(dat, Ur):
    
    '''
    This function projects data onto smaller subspace. Projection matrix defined by SVD
    
    Parameters:
    -----------
    dat: numpy array of all records
    Ur: projection matrix

    Returns:
    -------
    dat: projected numpy array of all records

    '''
  
    dat = np.dot(dat, Ur)
    return dat


def OutputPred(pred, header, pathV):

    '''
    Outputs prediction to a file
    
    Parameters:
    ----------
    pathV: specifies the output file
    pred: an array of class predictions
    header: column labels
 
    Returns:
    ----------
    none

    '''

    paths = ['predSVM.csv', 'predRF.csv']
    
    open_file_object = csv.writer(open(paths[pathV], "wb"))
    open_file_object.writerow(header)

    for i in xrange(0, len(pred)):
        open_file_object.writerow([i + 1, int(pred[i])])
