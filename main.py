from scipy.stats.morestats import Mean
import stats
import dimensionality_reduction_techniques as redTec
import numpy
import scipy.stats
import MVGclassifiers
import model_validation
import probability as prob
import logisticRegression
import matplotlib.pyplot as plt
import SVMClassifier
import gaussian_mixture_models 
import preprocessing
import validation_results
import evaluation_results

from numpy.random import permutation


import sklearn.preprocessing



def mrow(v):
    return v.reshape((1, v.size))

def mcol(v):
    return v.reshape((v.size, 1))

def load(fname):
    DList = []
    labelsList = []

    with open(fname) as f:
        for line in f:
            try:
                attrs = line.split(',')[0:11]
                attrs = mcol(numpy.array([float(i) for i in attrs]))
                label = int(line.split(',')[-1].strip())
                DList.append(attrs)
                labelsList.append(label)
            except:
                print("eccezione riga 22")
                pass

    return numpy.hstack(DList), numpy.array(labelsList, dtype=numpy.int32)

 

if __name__ == '__main__':

    DTR, LTR = load('Data/wine/Train.txt')
    DTE, LTE = load('Data/wine/Test.txt')

    ## DTR: Training Data  
    ## DTE: Evaluation Data
    ## LTR: Training Labels
    ## LTE: Evaluation Labels
    
    
    ## - compute statistics to analyse the data and the given features
    
    
    # plot histograms of the raw training dataset
    stats.plot_hist(DTR, LTR, "Stat/Hist/Raw")
    
    # plot histograms of the Z_normalized training dataset
    stats.plot_hist(preprocessing.Z_normalization(DTR), LTR, "Stat/Hist/Normalized")
    
    ## - gaussianize the features

    gaussianizedFeatures = preprocessing.gaussianization(DTR)
    stats.plot_hist(gaussianizedFeatures, LTR, "Stat/Hist/Gaussianized")
    
    
    # Plot scatters
    
    stats.plot_scatter(preprocessing.Z_normalization(DTR), LTR, "Raw")
    stats.plot_scatter(preprocessing.gaussianization(DTR), LTR, "Gaussianized")
    
    
    ## heat maps of the gaussianized features to show correlations between features
    stats.plot_heatmaps(preprocessing.gaussianization(DTR), LTR, "Stat/HeatMaps/Gaussianized")
    stats.plot_heatmaps(preprocessing.Z_normalization(DTR), LTR, "Stat/HeatMaps/Normalized")
    stats.plot_heatmaps(DTR, LTR, "Stat/HeatMaps/Raw")
    
    
    ##enstablish if data are balanced
    #Training
    n_high_qty = numpy.count_nonzero(LTR == 1)
    n_low_qty = numpy.count_nonzero(LTR == 0)
    print("train, high:", n_high_qty, "low: ", n_low_qty)
    ##-----> number of low qty >> number of high qty
    stats.bars_numsamples(n_high_qty, n_low_qty, "Training")
    #Test
    n_high_qty = numpy.count_nonzero(LTE == 1)
    n_low_qty = numpy.count_nonzero(LTE == 0)
    stats.bars_numsamples(n_high_qty, n_low_qty, "Test")
    print("test, high:", n_high_qty, "low: ", n_low_qty)
    
    
    ##choose k for k cross validation
    k = 5

    ##VALIDATION OF THE CLASSIFIERS : 
    validation_results.print_all(DTR, LTR, k)

    ##EVALUATION OF THE CLASSIFIERS :
    
    evaluation_results.print_all(DTR, LTR, DTE, LTE, k)
    


    
    
    

    