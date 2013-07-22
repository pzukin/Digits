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

    paths = ['predSVM.csv', 'predRF.csv', 'predRotSVM.csv']
    
    open_file_object = csv.writer(open(paths[pathV], "wb"))
    open_file_object.writerow(header)

    for i in xrange(0, len(pred)):
        open_file_object.writerow([i + 1, int(pred[i])])


def CalcRotFeatures(dat):

    '''
    This function calculates orientation independent features. 
    More specifically it takes into account how the digit is alligned
    relative to the axis of the image. 

    First we calculate the moment of inertia tensor to find the orientation
    of the digit. Then we calculate the ratio of the eigenvalues and other moments with
    respect to angular projections, in different radial bins.

    I Vectorized the calculation.

    Parameters:
    -----------
    dat: numpy array of all records

    Returns:
    --------
    datRot: various features that are orientation independent

    '''

    # mapping of index in record to x,y coordinates
    xAll = np.array( [i % 28 for i in xrange(0, 28**2)] )
    yAll = np.array( [28 - i / 28 for i in xrange(0, 28**2)] )

    # Define mass (weight) and center of mass
    weight = np.sum(dat, axis = 1)
    xCM = np.dot(xAll, dat.T) /  weight
    yCM = np.dot(yAll, dat.T) / weight

    # displacement in x and y of each pixel. radius of each pixel from center
    dx = np.zeros( (len(dat), 28**2), dtype = np.float64)
    dy = np.zeros( (len(dat), 28**2), dtype = np.float64)
    dr = np.zeros( (len(dat), 28**2), dtype = np.float64)

    # new feature array that will be returned
    # columns are: ratio of eigenvalues of Moment Intertia tensor, and moments of 
    # pixels wrt angular function (ie: sin(theta), cos(theta), sin(2*theta)...)
    # also splitting into different radial bins
    datRot = np.zeros( (len(dat), 16), dtype = np.float64)

    for i in xrange(0, 28**2):
        dx[:,i] = xAll[i] - xCM
        dy[:,i] = yAll[i] - yCM

    # useful if you want to split moments into radial bins
    dr = np.sqrt(dx**2 + dy**2)
    # define bins
    bins = np.linspace(0, np.max(dr) * 0.7, num = 4) 

    # coefficients of moment of inertia tensor
    a = np.sum(dx**2 * dat, axis = 1)
    b = np.sum(dx * dy * dat, axis = 1)
    d = np.sum(dy**2 * dat, axis = 1)

    # trace and determinant
    Tr = a + d
    Det = a * d - b * b

    # ratio of eigenvalues
    datRot[:,-1] = ( (Tr / 2.0 + np.sqrt(Tr**2 / 4.0 - Det) ) /
                    (Tr / 2.0 - np.sqrt(Tr**2 / 4.0 - Det) ) )

    # now moments. first define angle of principal axis

    # angle of rotation relative to vertical for each image
    eigVecY = Tr / 2.0 + np.sqrt(Tr**2 / 4.0 - Det) - a
    angle = np.arccos( eigVecY / np.sqrt(eigVecY**2 + b**2) )
    # takes into account clockwise or counter clockwise
    cc = np.ones(len(dat), dtype = np.float64) 
    mask = b < 0
    cc[mask] = -1.0
    mask = angle > (np.pi / 2.0)
    angle[mask] = np.pi - angle[mask]
    cc[mask] = -1.0 * cc[mask]
    angle = cc * angle

    # define angle at each pixel
    theta = np.arctan2(dy, dx)
    # now shift angles to be relative to principal axis
    theta = (theta.T + angle).T

    # now take moments for different radial bins
    for i in xrange(0, len(bins) - 1):

        mask = ( (dr[0,:] <= bins[i+1]) & (dr[0,:] > bins[i]) )
        datRot[:, i * 5] = ( np.sum(dat[:,mask], axis = 1) 
                                         / weight )
        datRot[:, 1 + i * 5] = ( np.sum(dat[:,mask] * np.sin(theta[:,mask]), axis = 1) 
                                 / weight )
        datRot[:, 2 + i * 5] = ( np.sum(dat[:,mask] * np.cos(theta[:,mask]), axis = 1) 
                                 / weight )
        datRot[:, 3 + i * 5] = ( np.sum(dat[:,mask] * np.sin(2.0 * theta[:,mask]), axis = 1) 
                                 / weight )
        datRot[:, 4 + i * 5] = ( np.sum(dat[:,mask] * np.cos(2.0 * theta[:,mask]), axis = 1) 
                                 / weight )
 
    return datRot





