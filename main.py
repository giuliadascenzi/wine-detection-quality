import stats
import dimensionality_reduction_techniques as redTec
import numpy
import scipy.stats
import classifiers
import model_evaluation
import probability as prob



def mcol(v):
    return v.reshape((v.size, 1))

def load(fname):
    DList = []
    labelsList = []

    with open(fname) as f:
        for line in f:
            try:
                attrs = line.split(',')[0:10]
                attrs = mcol(numpy.array([float(i) for i in attrs]))
                label = int(line.split(',')[-1].strip())
                DList.append(attrs)
                labelsList.append(label)
            except:
                print("eccezione riga 22")
                pass

    return numpy.hstack(DList), numpy.array(labelsList, dtype=numpy.int32)









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





if __name__ == '__main__':
    DTR, LTR = load('Data/wine/Train.txt')
    DTE, LTE = load('Data/wine/Test.txt')

    # DTR: Training Data
    # DTE: Evaluation Data
    # LTR: Training Labels
    # LTE: Evaluation Labels
    
    # compute statistics to analyse the data and the given features
    #stats.compute_stats(DTR, LTR, show_figures = True)

    #gaussianize the features
    gaussianizedFeatures = gaussianization(DTR)
    
    #stats.plot_hist(gaussianizedFeatures, LTR)


    #enstablish if data are balanced
    n_high_qty = numpy.count_nonzero(LTR == 1)
    n_low_qty = numpy.count_nonzero(LTR == 0)
    #-----> number of low qty >> number of high qty

    prior = 0.5
    cost_fn = 1
    cost_fp = 1
    classes_prior_probabilties = numpy.array([prior, 1-prior])

    #choose k for k cross validation
    k = 5

    #------------------------RAW FEATURES -----------------
    #Full_Cov 
    min_DCF_MVG = model_evaluation.k_cross_minDCF(DTR, LTR, k, classifiers.MVG_logLikelihoodRatios, prior , cost_fn, cost_fp)
    print("RAW FEATURES - min DCF MVG: ",min_DCF_MVG)  

    #Diag_Cov == Naive
    min_DCF_Diag_Cov = model_evaluation.k_cross_minDCF(DTR, LTR, k, classifiers.NAIVE_logLikelihoodRatios, prior , cost_fn, cost_fp)
    print("RAW FEATURES - min DCF MVG with Diag cov: ",min_DCF_Diag_Cov)

    #Tied
    min_DCF_Tied = model_evaluation.k_cross_minDCF(DTR, LTR, k, classifiers.TIED_logLikelihoodRatios, prior , cost_fn, cost_fp)
    print("RAW FEATURES - min DCF Tied MVG: ",min_DCF_Tied)

    #Tied Diag_Cov
    min_DCF_Tied_Diag_Cov = model_evaluation.k_cross_minDCF(DTR, LTR, k, classifiers.TIED_DIAG_COV_logLikelihoodRatios, prior , cost_fn, cost_fp)
    print("RAW FEATURES - min DCF Tied MVG with Diag Cov: ",min_DCF_Tied_Diag_Cov)


    #------------------------RAW FEATURES WITH PCA = 9 --------------------
    principal_components_9= redTec.PCA(DTR, 9)

    #Full_Cov 
    min_DCF_MVG = model_evaluation.k_cross_minDCF(principal_components_9, LTR, k, classifiers.MVG_logLikelihoodRatios, prior , cost_fn, cost_fp)
    print("RAW FEATURES with PCA = 9 - min DCF MVG: ",min_DCF_MVG)  

    #Diag_Cov == Naive
    min_DCF_Diag_Cov = model_evaluation.k_cross_minDCF(principal_components_9, LTR, k, classifiers.NAIVE_logLikelihoodRatios, prior , cost_fn, cost_fp)
    print("RAW FEATURES with PCA = 9 - min DCF MVG with Diag cov: ",min_DCF_Diag_Cov)

    #Tied
    min_DCF_Tied = model_evaluation.k_cross_minDCF(principal_components_9, LTR, k, classifiers.TIED_logLikelihoodRatios, prior , cost_fn, cost_fp)
    print("RAW FEATURES with PCA = 9 - min DCF Tied MVG: ",min_DCF_Tied)

    #Tied Diag_Cov
    min_DCF_Tied_Diag_Cov = model_evaluation.k_cross_minDCF(principal_components_9, LTR, k, classifiers.TIED_DIAG_COV_logLikelihoodRatios, prior , cost_fn, cost_fp)
    print("RAW FEATURES with PCA = 9 - min DCF Tied MVG with Diag Cov: ",min_DCF_Tied_Diag_Cov)



    #------------------------RAW FEATURES WITH PCA = 8 --------------------
    principal_components_8= redTec.PCA(DTR, 8)

    #Full_Cov 
    min_DCF_MVG = model_evaluation.k_cross_minDCF(principal_components_8, LTR, k, classifiers.MVG_logLikelihoodRatios, prior , cost_fn, cost_fp)
    print("RAW FEATURES with PCA = 8 - min DCF MVG: ",min_DCF_MVG)  

    #Diag_Cov == Naive
    min_DCF_Diag_Cov = model_evaluation.k_cross_minDCF(principal_components_8, LTR, k, classifiers.NAIVE_logLikelihoodRatios, prior , cost_fn, cost_fp)
    print("RAW FEATURES with PCA = 8 - min DCF MVG with Diag cov: ",min_DCF_Diag_Cov)

    #Tied
    min_DCF_Tied = model_evaluation.k_cross_minDCF(principal_components_8, LTR, k, classifiers.TIED_logLikelihoodRatios, prior , cost_fn, cost_fp)
    print("RAW FEATURES with PCA = 8 - min DCF Tied MVG: ",min_DCF_Tied)

    #Tied Diag_Cov
    min_DCF_Tied_Diag_Cov = model_evaluation.k_cross_minDCF(principal_components_8, LTR, k, classifiers.TIED_DIAG_COV_logLikelihoodRatios, prior , cost_fn, cost_fp)
    print("RAW FEATURES with PCA = 8 - min DCF Tied MVG with Diag Cov: ",min_DCF_Tied_Diag_Cov)


    #--------------- GAUSSIANIZED FEATURES-------------------------

    #Full_Cov 
    min_DCF_MVG = model_evaluation.k_cross_minDCF(gaussianizedFeatures, LTR, k, classifiers.MVG_logLikelihoodRatios, prior , cost_fn, cost_fp)
    print("GAUSSIANIZED FEATURES - min DCF MVG: ",min_DCF_MVG)  

    #Diag_Cov == Naive
    min_DCF_Diag_Cov = model_evaluation.k_cross_minDCF(gaussianizedFeatures, LTR, k, classifiers.NAIVE_logLikelihoodRatios, prior , cost_fn, cost_fp)
    print("GAUSSIANIZED FEATURES - min DCF MVG with Diag cov: ",min_DCF_Diag_Cov)

    #Tied
    min_DCF_Tied = model_evaluation.k_cross_minDCF(gaussianizedFeatures, LTR, k, classifiers.TIED_logLikelihoodRatios, prior , cost_fn, cost_fp)
    print("GAUSSIANIZED FEATURES - min DCF Tied MVG: ",min_DCF_Tied)

    #Tied Diag_Cov
    min_DCF_Tied_Diag_Cov = model_evaluation.k_cross_minDCF(gaussianizedFeatures, LTR, k, classifiers.TIED_DIAG_COV_logLikelihoodRatios, prior , cost_fn, cost_fp)
    print("GAUSSIANIZED FEATURES - min DCF Tied MVG with Diag Cov: ",min_DCF_Tied_Diag_Cov)


    #------------------------GAUSSIANIZED FEATURES WITH PCA = 9 --------------------
    gaussianized_principal_components_9= redTec.PCA(gaussianizedFeatures, 9)

    #Full_Cov 
    min_DCF_MVG = model_evaluation.k_cross_minDCF(gaussianized_principal_components_9, LTR, k, classifiers.MVG_logLikelihoodRatios, prior , cost_fn, cost_fp)
    print("GAUSSIANIZED FEATURES with PCA = 9 - min DCF MVG: ",min_DCF_MVG)  

    #Diag_Cov == Naive
    min_DCF_Diag_Cov = model_evaluation.k_cross_minDCF(gaussianized_principal_components_9, LTR, k, classifiers.NAIVE_logLikelihoodRatios, prior , cost_fn, cost_fp)
    print("GAUSSIANIZED FEATURES with PCA = 9 - min DCF MVG with Diag cov: ",min_DCF_Diag_Cov)

    #Tied
    min_DCF_Tied = model_evaluation.k_cross_minDCF(gaussianized_principal_components_9, LTR, k, classifiers.TIED_logLikelihoodRatios, prior , cost_fn, cost_fp)
    print("GAUSSIANIZED FEATURES with PCA = 9 - min DCF Tied MVG: ",min_DCF_Tied)

    #Tied Diag_Cov
    min_DCF_Tied_Diag_Cov = model_evaluation.k_cross_minDCF(gaussianized_principal_components_9, LTR, k, classifiers.TIED_DIAG_COV_logLikelihoodRatios, prior , cost_fn, cost_fp)
    print("GAUSSIANIZED FEATURES with PCA = 9 - min DCF Tied MVG with Diag Cov: ",min_DCF_Tied_Diag_Cov)



    #------------------------GAUSSIANIZED FEATURES WITH PCA = 8 --------------------
    gaussianized_principal_components_8= redTec.PCA(gaussianizedFeatures, 8)

    #Full_Cov 
    min_DCF_MVG = model_evaluation.k_cross_minDCF(gaussianized_principal_components_8, LTR, k, classifiers.MVG_logLikelihoodRatios, prior , cost_fn, cost_fp)
    print("GAUSSIANIZED FEATURES with PCA = 8 - min DCF MVG: ",min_DCF_MVG)  

    #Diag_Cov == Naive
    min_DCF_Diag_Cov = model_evaluation.k_cross_minDCF(gaussianized_principal_components_8, LTR, k, classifiers.NAIVE_logLikelihoodRatios, prior , cost_fn, cost_fp)
    print("GAUSSIANIZED FEATURES with PCA = 8 - min DCF MVG with Diag cov: ",min_DCF_Diag_Cov)

    #Tied
    min_DCF_Tied = model_evaluation.k_cross_minDCF(gaussianized_principal_components_8, LTR, k, classifiers.TIED_logLikelihoodRatios, prior , cost_fn, cost_fp)
    print("GAUSSIANIZED FEATURES with PCA = 8 - min DCF Tied MVG: ",min_DCF_Tied)

    #Tied Diag_Cov
    min_DCF_Tied_Diag_Cov = model_evaluation.k_cross_minDCF(gaussianized_principal_components_8, LTR, k, classifiers.TIED_DIAG_COV_logLikelihoodRatios, prior , cost_fn, cost_fp)
    print("GAUSSIANIZED FEATURES with PCA = 8 - min DCF Tied MVG with Diag Cov: ",min_DCF_Tied_Diag_Cov)
    #stats.plot_scatter(principal_components,LTR)
    #linear_discriminants = redTec.LDA(DTR,LTR, 1)
    #redTec.plotLDA(linear_discriminants, LTR, "Applied LDA")


   

           

