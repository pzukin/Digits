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

def plotHistEig(dat, tar):

    '''
    This function calculates moment of inertia tensor for each record.
    Then takes the ratio of the eigenvalues and plots a histogram 
    for each digit.

    Parameters:
    -----------
    dat: numpy array of all records
    tar: numpy array with class labels for all records

    '''

    # generate figure
    fig = plt.figure(1, figsize = (10, 10))
    ax = fig.add_subplot(1, 1, 1)
    lsv=['-','-','-','-','-','-','-','--','--','--']
    prop = matplotlib.font_manager.FontProperties(size=10.5)

    # mapping of index in record to x,y coordinates
    xAll = np.array( [i % 28 for i in xrange(0, 28**2)] )
    yAll = np.array( [28 - i / 28 for i in xrange(0, 28**2)] )

    for j in xrange(0,10):

        # pick out all records in a particular class
        mask = tar == np.float64(j)
        print "    Digit:", j, mask.sum(), len(dat)

        # vectorizing (doing it individually takes very long)
        weight = np.sum(dat[mask,:], axis = 1)
        xCM = np.dot(xAll, dat[mask].T) /  weight
        yCM = np.dot(yAll, dat[mask].T) / weight

        dx = np.zeros( (mask.sum(), 28**2), dtype = np.float64)
        dy = np.zeros( (mask.sum(), 28**2), dtype = np.float64)

        for i in xrange(0, 28**2):
            dx[:,i] = xAll[i] - xCM
            dy[:,i] = yAll[i] - yCM

        # coefficients of moment of inertia tensor
        a = np.dot( dx**2, dat[mask].T ).diagonal() 
        b = np.dot( dx * dy, dat[mask].T ).diagonal() 
        d = np.dot( dy**2, dat[mask].T ).diagonal()

        # trace and determinant
        Tr = a + d
        Det = a * d - b * b

        # ratio of eigenvalues
        vals = ( (Tr / 2.0 + np.sqrt(Tr**2 / 4.0 - Det) ) / 
                (Tr / 2.0 - np.sqrt(Tr**2 / 4.0 - Det) ) )

        #print rat[:5]

        # now make histogram and plot
        hist, bin_edges = np.histogram(vals, bins = 30, density=True)
        ax.plot( 0.5 * (bin_edges[1:] + bin_edges[:-1]),
                 hist, lw = 2.0, label = str(j), ls = lsv[j])

    ax.axis([0, 40, 0, 0.8])
    ax.semilogx()
    plt.legend(loc='upper right',prop=prop)
    plt.savefig('Eig.pdf', bbox_inches = 'tight')
    fig.clear()

def main():

    # first read in data. tar == class label. dat == data
    print
    print "Reading Data..."
    dat, tar = utils.ReadDat("train.csv", 1)
    print "Done Reading.\n"

    print "Investigating moment of Inertia:"
    plotHistEig(dat, tar)

    print 'Done'


if __name__ == '__main__':
    main()
