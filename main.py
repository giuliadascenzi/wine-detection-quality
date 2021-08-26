import stats
import dimensionality_reduction_techniques as redTec
import numpy
import scipy.stats
import MVGclassifiers
import model_evaluation
import probability as prob
import logisticRegression



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

def print_table_MVG_classifiers_minDCF(DTR, prior, cost_fn, cost_fp, k):

    def MVG_Classifiers_minDCF(data):
        #Full_Cov 
        min_DCF_MVG = model_evaluation.singleFold_minDCF(data, LTR, MVGclassifiers.MVG_logLikelihoodRatios, prior , cost_fn, cost_fp)
        print("[Single Fold] -  MVG: ",min_DCF_MVG)  
        min_DCF_MVG = model_evaluation.k_cross_minDCF(data, LTR, k, MVGclassifiers.MVG_logLikelihoodRatios, prior , cost_fn, cost_fp)
        print("[5-Folds]  -  MVG: ",min_DCF_MVG)  

        #Diag_Cov == Naive
        min_DCF_Diag_Cov = model_evaluation.singleFold_minDCF(data, LTR, MVGclassifiers.NAIVE_logLikelihoodRatios, prior , cost_fn, cost_fp)
        print("[Single Fold]  - MVG with Diag cov: ",min_DCF_Diag_Cov)
        min_DCF_Diag_Cov = model_evaluation.k_cross_minDCF(data, LTR,k, MVGclassifiers.NAIVE_logLikelihoodRatios, prior , cost_fn, cost_fp)
        print("[5- Fold] - MVG with Diag cov: ",min_DCF_Diag_Cov)

        #Tied
        min_DCF_Tied = model_evaluation.singleFold_minDCF(data, LTR, MVGclassifiers.TIED_logLikelihoodRatios, prior , cost_fn, cost_fp)
        print("[Single Fold] - Tied MVG: ",min_DCF_Tied)
        min_DCF_Tied = model_evaluation.k_cross_minDCF(DTR, LTR,k, MVGclassifiers.TIED_logLikelihoodRatios, prior , cost_fn, cost_fp)
        print("[5- Fold] - Tied MVG: ",min_DCF_Tied)

        #Tied Diag_Cov
        min_DCF_Tied_Diag_Cov = model_evaluation.singleFold_minDCF(DTR, LTR, MVGclassifiers.TIED_DIAG_COV_logLikelihoodRatios, prior , cost_fn, cost_fp)
        print("[Single Fold] - Tied MVG with Diag Cov: ",min_DCF_Tied_Diag_Cov)
        min_DCF_Tied_Diag_Cov = model_evaluation.singleFold_minDCF(DTR, LTR, MVGclassifiers.TIED_DIAG_COV_logLikelihoodRatios, prior , cost_fn, cost_fp)
        print("[5 Fold] - Tied MVG with Diag Cov: ",min_DCF_Tied_Diag_Cov)

        print()

    
    #------------------------RAW FEATURES -----------------
    print("*** minDCF - RAW FEATURES - NO PCA ***")
    MVG_Classifiers_minDCF(DTR)
    
    #------------------------RAW FEATURES WITH PCA = 9 --------------------
    
    principal_components9= redTec.PCA(DTR, 9)
    print("*** minDCF - RAW FEATURES -  PCA (m=9) ***")
    MVG_Classifiers_minDCF(principal_components9)       


    #------------------------RAW FEATURES WITH PCA = 8 --------------------
    
    principal_components8= redTec.PCA(DTR, 8)
    print("*** minDCF - RAW FEATURES -  PCA (m=8) ***")
    MVG_Classifiers_minDCF(principal_components8)    


    #--------------- GAUSSIANIZED FEATURES-------------------------

    print("*** minDCF - GAUSSIANIZED FEATURES - NO PCA ***")
    MVG_Classifiers_minDCF(gaussianizedFeatures)

    #------------------------GAUSSIANIZED FEATURES WITH PCA = 9 --------------------
    print("*** minDCF - GAUSSIANIZED FEATURES -  PCA m=9 ***")
    gaussianized_principal_components_9= redTec.PCA(gaussianizedFeatures, 9)
    MVG_Classifiers_minDCF(gaussianized_principal_components_9)     
    
    #------------------------GAUSSIANIZED FEATURES WITH PCA = 8 --------------------
    print("*** minDCF - GAUSSIANIZED FEATURES -  PCA m=8 ***")
    gaussianized_principal_components_8= redTec.PCA(gaussianizedFeatures, 8)
    MVG_Classifiers_minDCF(gaussianized_principal_components_8)



if __name__ == '__main__':

    DTR, LTR = load('Data/wine/Train.txt')
    DTE, LTE = load('Data/wine/Test.txt')

    ## DTR: Training Data
    ## DTE: Evaluation Data
    ## LTR: Training Labels
    ## LTE: Evaluation Labels
    
    ## - compute statistics to analyse the data and the given features
    #stats.compute_stats(DTR, LTR, show_figures = True)

    ## - gaussianize the features
    gaussianizedFeatures = gaussianization(DTR)
    stats.plot_hist(gaussianizedFeatures, LTR)


    ##enstablish if data are balanced
    n_high_qty = numpy.count_nonzero(LTR == 1)
    n_low_qty = numpy.count_nonzero(LTR == 0)
    ##-----> number of low qty >> number of high qty
    
    ## for the balanced application:
    prior = 0.5
    cost_fn = 1
    cost_fp = 1
    classes_prior_probabilties = numpy.array([prior, 1-prior])

    ##choose k for k cross validation
    k = 5

    ##EVAULATION OF THE CLASSIFIERS : 
    ### -- MVG CLASSIFIERS
    '''
    print("------> pi = 0.5")
    print_table_MVG_classifiers_minDCF(DTR, prior=0.5, cost_fn=1, cost_fp=1, k=k)
    print()
    print("------> pi = 0.9")
    print_table_MVG_classifiers_minDCF(DTR, prior=0.9, cost_fn=1, cost_fp=1, k=k)
    print()
    print("------> pi = 0.1")
    print_table_MVG_classifiers_minDCF(DTR, prior=0.1, cost_fn=1, cost_fp=1, k=k)
    print()

    '''

    ### -- LOGISTIC REGRESSION

    scores = logisticRegression.LR_logLikelihoodRatios(DTR, LTR, DTE, lam=1, pi_T=0.5)
    print(scores)


    
    #stats.plot_scatter(principal_components,LTR)
    #linear_discriminants = redTec.LDA(DTR,LTR, 1)
    #redTec.plotLDA(linear_discriminants, LTR, "Applied LDA")


   

           

