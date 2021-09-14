import numpy
import scipy.stats


def mrow(v):
    return v.reshape((1, v.size))

def mcol(v):
    return v.reshape((v.size, 1))

def Z_normalization(D): #centering and scaling to unit variance
    return (D-mcol(D.mean(1)))/mcol(numpy.std(D, axis=1))


def gaussianization(D):
    N = D.shape[1]
    r = numpy.zeros(D.shape)
    for k in range(D.shape[0]):
        featureVector = D[k,:]
        ranks = scipy.stats.rankdata(featureVector, method='min') -1
        r[k,:] = ranks

    r = (r + 1)/(N+2)
    gaussianizedFeatures = scipy.stats.norm.ppf(r)
    return gaussianizedFeatures

'''
def gaussianizationEval(D, X):
    N = D.shape[1]
    r = numpy.zeros(X.shape)
    for k in range(X.shape[0]):
        featureVectorD = D[k,:]
        for i in range (X.shape[1]):
            ranks = scipy.stats.rankdata(numpy.append(featureVectorD, X[k][i]), method='min') -1
            r[k,i] = ranks[-1]

    r = (r + 1)/(N+2)
    gaussianizedFeatures = scipy.stats.norm.ppf(r)
    return gaussianizedFeatures
'''

def gaussianizationEval(DTR, DEV):
    D = numpy.concatenate((DTR, DEV), axis=1)
    r = numpy.zeros(DEV.shape)
    for k in range(DEV.shape[0]):
        featureVector_all = D[k,:]
        featureVector_DEV = DEV[k,:]

        EV_ranks = scipy.stats.rankdata(featureVector_DEV, method="min") -1
        ALL_ranks = scipy.stats.rankdata(featureVector_all, method='min') -1
        ALL_ranks = ALL_ranks[DTR.shape[1]:]
        
        r[k,:] = ALL_ranks - EV_ranks

    r = (r + 1)/(DTR.shape[1]+2)
    gaussianizedFeatures = scipy.stats.norm.ppf(r)
    return gaussianizedFeatures


