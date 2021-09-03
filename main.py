from scipy.stats.morestats import Mean
import stats
import dimensionality_reduction_techniques as redTec
import numpy
import scipy.stats
import MVGclassifiers
import model_evaluation
import probability as prob
import logisticRegression
import matplotlib.pyplot as plt
import SVMClassifier
import gaussian_mixture_models 

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
                attrs = line.split(',')[0:10]
                attrs = mcol(numpy.array([float(i) for i in attrs]))
                label = int(line.split(',')[-1].strip())
                DList.append(attrs)
                labelsList.append(label)
            except:
                print("eccezione riga 22")
                pass

    return numpy.hstack(DList), numpy.array(labelsList, dtype=numpy.int32)


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

def print_table_MVG_classifiers_minDCF(DTR, prior, cost_fn, cost_fp, k):

    def MVG_Classifiers_minDCF(data):
        #Full_Cov 
        min_DCF_MVG,_ = model_evaluation.singleFold_DCF(data, LTR, MVGclassifiers.MVG_logLikelihoodRatios, prior , cost_fn, cost_fp)
        print("[Single Fold] -  MVG: ",min_DCF_MVG)  
        min_DCF_MVG,_,_ = model_evaluation.k_cross_DCF(data, LTR, k, MVGclassifiers.MVG_logLikelihoodRatios, prior , cost_fn, cost_fp)
        print("[5-Folds]  -  MVG: ",min_DCF_MVG)  

        #Diag_Cov == Naive
        min_DCF_Diag_Cov,_ = model_evaluation.singleFold_DCF(data, LTR, MVGclassifiers.NAIVE_logLikelihoodRatios, prior , cost_fn, cost_fp)
        print("[Single Fold]  - MVG with Diag cov: ",min_DCF_Diag_Cov)
        min_DCF_Diag_Cov,_,_ = model_evaluation.k_cross_DCF(data, LTR,k, MVGclassifiers.NAIVE_logLikelihoodRatios, prior , cost_fn, cost_fp)
        print("[5- Fold] - MVG with Diag cov: ",min_DCF_Diag_Cov)

        #Tied
        min_DCF_Tied,_ = model_evaluation.singleFold_DCF(data, LTR, MVGclassifiers.TIED_logLikelihoodRatios, prior , cost_fn, cost_fp)
        print("[Single Fold] - Tied MVG: ",min_DCF_Tied)
        min_DCF_Tied,_,_ = model_evaluation.k_cross_DCF(data, LTR,k, MVGclassifiers.TIED_logLikelihoodRatios, prior , cost_fn, cost_fp)
        print("[5- Fold] - Tied MVG: ",min_DCF_Tied)

        #Tied Diag_Cov
        min_DCF_Tied_Diag_Cov,_ = model_evaluation.singleFold_DCF(data, LTR, MVGclassifiers.TIED_DIAG_COV_logLikelihoodRatios, prior , cost_fn, cost_fp)
        print("[Single Fold] - Tied MVG with Diag Cov: ",min_DCF_Tied_Diag_Cov)
        min_DCF_Tied_Diag_Cov,_,_ = model_evaluation.k_cross_DCF(data, LTR, k,  MVGclassifiers.TIED_DIAG_COV_logLikelihoodRatios, prior , cost_fn, cost_fp)
        print("[5 Fold] - Tied MVG with Diag Cov: ",min_DCF_Tied_Diag_Cov)

        print()

    #!!! normalization is important before PCA
    normalized_data = Z_normalization(DTR)


    #------------------------RAW FEATURES (normalized) -----------------
    print("*** minDCF - RAW (normalized) FEATURES - NO PCA ***")
    MVG_Classifiers_minDCF(normalized_data)
    

    #------------------------RAW FEATURES (normalized) WITH PCA = 9 --------------------
    principal_components9= redTec.PCA(normalized_data, 9)
    print("*** minDCF - RAW (normalized) FEATURES -  PCA (m=9) ***")
    MVG_Classifiers_minDCF(principal_components9)       


    #------------------------RAW FEATURES (normalized) WITH PCA = 8 --------------------
    
    principal_components8= redTec.PCA(normalized_data, 8)
    print("*** minDCF - RAW (normalized) FEATURES -  PCA (m=8) ***")
    MVG_Classifiers_minDCF(principal_components8)    


    ## Z --> PCA --> GAUSS
    #--------------- GAUSSIANIZED FEATURES-------------------------
    gaussianizedFeatures = gaussianization(DTR)
    print("*** minDCF - GAUSSIANIZED FEATURES - NO PCA ***")
    MVG_Classifiers_minDCF(gaussianizedFeatures)


    #------------------------GAUSSIANIZED FEATURES WITH PCA = 9 --------------------
    print("*** minDCF - GAUSSIANIZED FEATURES -  PCA m=9 ***")
    principal_components9= redTec.PCA(normalized_data, 9)
    gaussianized_principal_components_9 = gaussianization(principal_components9)
    MVG_Classifiers_minDCF(gaussianized_principal_components_9)     


    #------------------------GAUSSIANIZED FEATURES WITH PCA = 8 --------------------
    print("*** minDCF - GAUSSIANIZED FEATURES -  PCA m=8 ***")
    principal_components8= redTec.PCA(normalized_data, 8)
    gaussianized_principal_components_8= gaussianization(principal_components8)
    MVG_Classifiers_minDCF(gaussianized_principal_components_8)


def print_table_LR_minDCF(DTR, LTR, prior, cost_fn, cost_fp, k):

    def LR_minDCF(data):
            
            lam = 10**(-7)
            pi_T = 0.5
            min_DCF_LR,_,_ = model_evaluation.k_cross_DCF(data, LTR, k, logisticRegression.LR_logLikelihoodRatios, prior , cost_fn, cost_fp, [lam, pi_T])
            print("[5-Folds]  -  lam = 10^-7, pi_T = 0.5: ",min_DCF_LR)  

            pi_T = 0.1
            min_DCF_LR,_,_ = model_evaluation.k_cross_DCF(data, LTR, k, logisticRegression.LR_logLikelihoodRatios, prior , cost_fn, cost_fp, [lam, pi_T])
            print("[5-Folds]  -  lam = 10^-7, pi_T = 0.1: ",min_DCF_LR)

            pi_T = 0.9
            min_DCF_LR,_,_ = model_evaluation.k_cross_DCF(data, LTR, k, logisticRegression.LR_logLikelihoodRatios, prior , cost_fn, cost_fp, [lam, pi_T])
            print("[5-Folds]  -  lam = 10^-7, pi_T = 0.9: ",min_DCF_LR)
            

            N = LTR.size #tot number of samples
            n_T = (1*(LTR==1)).sum() #num of samples belonging to the true class
            pi_emp_T = n_T / N

            pi_T = pi_emp_T
            
            min_DCF_LR,_,_ = model_evaluation.k_cross_DCF(data, LTR, k, logisticRegression.LR_logLikelihoodRatios, prior , cost_fn, cost_fp, [lam, pi_T])
            print("[5-Folds]  -  lam = 10^-7, pi_T = pi_emp_T: ",min_DCF_LR)

            print()

    
    #------------------------RAW FEATURES -----------------
    print("*** minDCF - RAW FEATURES ***")
    LR_minDCF(Z_normalization(DTR))

    #--------------- GAUSSIANIZED FEATURES-------------------------
    gaussianizedFeatures = gaussianization(DTR)

    print("*** minDCF - GAUSSIANIZED FEATURES  ***")
    LR_minDCF(gaussianizedFeatures)



def print_graphs_LR_lambdas(DTR, LTR,  k):
    def oneGraphSingleFold( data, prior, cost_fn, cost_fp, pi_T):
        print("working on single fold prior = ", prior)
        exps = numpy.linspace(-8,5, 14)
        lambdas = 10** exps
        minDCFs = 0 * exps
        for i in range (lambdas.size):
            lam= lambdas[i]
            minDCFs[i],_ = model_evaluation.singleFold_DCF(data, LTR, logisticRegression.LR_logLikelihoodRatios, prior , cost_fn, cost_fp, [lam, pi_T])
        
        lb = "minDCF (prior="+ str(prior)+ " )"
        print("DONE")
        plt.plot(lambdas, minDCFs, label=lb)
        plt.legend()
        

    def oneGraphKFold(data, prior, cost_fn, cost_fp, pi_T):
        print("working on k fold prior = ", prior)
        exps = numpy.linspace(-8,5, 14)
        lambdas = 10** exps
        minDCFs = 0 * exps
        k=5
        for i in range (lambdas.size):
            lam= lambdas[i]
            minDCFs[i],_,_ = model_evaluation.k_cross_DCF(data, LTR,k, logisticRegression.LR_logLikelihoodRatios, prior , cost_fn, cost_fp, [lam, pi_T])
        
        lb = "minDCF (prior="+ str(prior)+ " )"
        print("DONE")
        plt.plot(lambdas, minDCFs, label=lb)
        plt.legend()

    normalizedFeatures = Z_normalization(DTR)
    gaussianizedFeatures = gaussianization(DTR)

    plt.figure()
    print("+++++++++++++++++++++++++++++++++++")
    print("Raw features, single fold")
    plt.title("Raw features, single fold")
    plt.xscale('log')
    plt.xlabel("lambda")
    plt.ylabel("minDCFs")
    oneGraphSingleFold(normalizedFeatures, prior=0.5, cost_fn=1, cost_fp=1, pi_T=0.5)
    oneGraphSingleFold(normalizedFeatures, prior=0.1, cost_fn=1, cost_fp=1, pi_T=0.5)
    oneGraphSingleFold(normalizedFeatures, prior=0.9, cost_fn=1, cost_fp=1, pi_T=0.5)
    plt.savefig('Graph/LR/linear/singleFoldRAW.png' )

    
    plt.figure()
    print("+++++++++++++++++++++++++++++++++++")
    print("Gaussianized features, single fold")
    plt.title("Gaussianized features, single fold")
    plt.xscale('log')
    plt.xlabel("lambda")
    plt.ylabel("minDCFs")
    oneGraphSingleFold(gaussianizedFeatures, prior=0.5, cost_fn=1, cost_fp=1, pi_T=0.5)
    oneGraphSingleFold(gaussianizedFeatures, prior=0.1, cost_fn=1, cost_fp=1, pi_T=0.5)
    oneGraphSingleFold(gaussianizedFeatures, prior=0.9, cost_fn=1, cost_fp=1, pi_T=0.5)
    plt.savefig('Graph/LR/linear/singleFoldGauss.png' )

    plt.figure()
    print("+++++++++++++++++++++++++++++++++++")
    print("Raw features, 5 fold")
    plt.title("Raw features, 5 fold")
    plt.xscale('log')
    plt.xlabel("lambda")
    plt.ylabel("minDCFs")
    oneGraphKFold(normalizedFeatures, prior=0.5, cost_fn=1, cost_fp=1, pi_T=0.5)
    oneGraphKFold(normalizedFeatures, prior=0.1, cost_fn=1, cost_fp=1, pi_T=0.5)
    oneGraphKFold(normalizedFeatures, prior=0.9, cost_fn=1, cost_fp=1, pi_T=0.5)
    plt.savefig('Graph/LR/linear/5FoldRAW.png' )

    plt.figure()
    print("+++++++++++++++++++++++++++++++++++")
    print("Gaussianized features, 5 fold")
    plt.title("Gaussianized features, 5 fold")
    plt.xscale('log')
    plt.xlabel("lambda")
    plt.ylabel("minDCFs")
    oneGraphKFold(gaussianizedFeatures, prior=0.5, cost_fn=1, cost_fp=1, pi_T=0.5)
    oneGraphKFold(gaussianizedFeatures, prior=0.1, cost_fn=1, cost_fp=1, pi_T=0.5)
    oneGraphKFold(gaussianizedFeatures, prior=0.9, cost_fn=1, cost_fp=1, pi_T=0.5)
    plt.savefig('Graph/LR/linear/5FoldGauss.png' )
    

    #plt.show()

    


def print_graphs_quadratic_LR_lambdas(DTR, LTR,  k):
    def oneGraphSingleFold( data, prior, cost_fn, cost_fp, pi_T):
        print("working on single fold prior = ", prior)
        exps = numpy.linspace(-8,5, 14)
        lambdas = 10** exps
        minDCFs = 0 * exps
        for i in range (lambdas.size):
            lam= lambdas[i]
            minDCFs[i],_ = model_evaluation.singleFold_DCF(data, LTR, logisticRegression.Quadratic_LR_logLikelihoodRatios, prior , cost_fn, cost_fp, [lam, pi_T])
        
        lb = "minDCF (prior="+ str(prior)+ " )"
        print("DONE")
        plt.plot(lambdas, minDCFs, label=lb)
        plt.legend()
        

    def oneGraphKFold(data, prior, cost_fn, cost_fp, pi_T):
        print("working on k fold prior = ", prior)
        exps = numpy.linspace(-8,5, 14)
        lambdas = 10** exps
        minDCFs = 0 * exps
        k=5
        for i in range (lambdas.size):
            lam= lambdas[i]
            minDCFs[i],_,_ = model_evaluation.k_cross_DCF(data, LTR,k, logisticRegression.Quadratic_LR_logLikelihoodRatios, prior , cost_fn, cost_fp, [lam, pi_T])
        
        lb = "minDCF (prior="+ str(prior)+ " )"
        print("DONE")
        plt.plot(lambdas, minDCFs, label=lb)
        plt.legend()

    normalizedFeatures = Z_normalization(DTR)
    gaussianizedFeatures = gaussianization(DTR)

    plt.figure()
    print("+++++++++++++++++++++++++++++++++++")
    print("Raw features, single fold")
    plt.title("Raw features, single fold")
    plt.xscale('log')
    plt.xlabel("lambda")
    plt.ylabel("minDCFs")
    oneGraphSingleFold(normalizedFeatures, prior=0.5, cost_fn=1, cost_fp=1, pi_T=0.5)
    oneGraphSingleFold(normalizedFeatures, prior=0.1, cost_fn=1, cost_fp=1, pi_T=0.5)
    oneGraphSingleFold(normalizedFeatures, prior=0.9, cost_fn=1, cost_fp=1, pi_T=0.5)
    plt.savefig('Graph/LR/quadratic/singleFoldRAW.png' )

    
    plt.figure()
    print("+++++++++++++++++++++++++++++++++++")
    print("Gaussianized features, single fold")
    plt.title("Gaussianized features, single fold")
    plt.xscale('log')
    plt.xlabel("lambda")
    plt.ylabel("minDCFs")
    oneGraphSingleFold(gaussianizedFeatures, prior=0.5, cost_fn=1, cost_fp=1, pi_T=0.5)
    oneGraphSingleFold(gaussianizedFeatures, prior=0.1, cost_fn=1, cost_fp=1, pi_T=0.5)
    oneGraphSingleFold(gaussianizedFeatures, prior=0.9, cost_fn=1, cost_fp=1, pi_T=0.5)
    plt.savefig('Graph/LR/quadratic/singleFoldGauss.png' )

    plt.figure()
    print("+++++++++++++++++++++++++++++++++++")
    print("Raw features, 5 fold")
    plt.title("Raw features, 5 fold")
    plt.xscale('log')
    plt.xlabel("lambda")
    plt.ylabel("minDCFs")
    oneGraphKFold(normalizedFeatures, prior=0.5, cost_fn=1, cost_fp=1, pi_T=0.5)
    oneGraphKFold(normalizedFeatures, prior=0.1, cost_fn=1, cost_fp=1, pi_T=0.5)
    oneGraphKFold(normalizedFeatures, prior=0.9, cost_fn=1, cost_fp=1, pi_T=0.5)
    plt.savefig('Graph/LR/quadratic/5FoldRAW.png' )

    plt.figure()
    print("+++++++++++++++++++++++++++++++++++")
    print("Gaussianized features, 5 fold")
    plt.title("Gaussianized features, 5 fold")
    plt.xscale('log')
    plt.xlabel("lambda")
    plt.ylabel("minDCFs")
    oneGraphKFold(gaussianizedFeatures, prior=0.5, cost_fn=1, cost_fp=1, pi_T=0.5)
    oneGraphKFold(gaussianizedFeatures, prior=0.1, cost_fn=1, cost_fp=1, pi_T=0.5)
    oneGraphKFold(gaussianizedFeatures, prior=0.9, cost_fn=1, cost_fp=1, pi_T=0.5)
    plt.savefig('Graph/LR/quadratic/5FoldGauss.png' )
    
    
    #plt.show()

def print_table_Quadratic_LR_minDCF(DTR, LTR, prior, cost_fn, cost_fp, k):

    def Quad_LR_minDCF(data):
            lam = 10**(-7)
            '''
            pi_T = 0.5
            min_DCF_LR,_,_ = model_evaluation.k_cross_DCF(data, LTR, k, logisticRegression.Quadratic_LR_logLikelihoodRatios, prior , cost_fn, cost_fp, [lam, pi_T])
            print("[5-Folds]  -  lam = 10^-7, pi_T = 0.5: ",min_DCF_LR)  

            pi_T = 0.1
            min_DCF_LR,_,_ = model_evaluation.k_cross_DCF(data, LTR, k, logisticRegression.Quadratic_LR_logLikelihoodRatios, prior , cost_fn, cost_fp, [lam, pi_T])
            print("[5-Folds]  -  lam = 10^-7, pi_T = 0.1: ",min_DCF_LR)

            pi_T = 0.9
            min_DCF_LR,_,_ = model_evaluation.k_cross_DCF(data, LTR, k, logisticRegression.Quadratic_LR_logLikelihoodRatios, prior , cost_fn, cost_fp, [lam, pi_T])
            print("[5-Folds]  -  lam = 10^-7, pi_T = 0.9: ",min_DCF_LR)
            '''

            N = LTR.size #tot number of samples
            n_T = (1*(LTR==1)).sum() #num of samples belonging to the true class
            pi_emp_T = n_T / N

            pi_T = pi_emp_T
            min_DCF_LR,_,_ = model_evaluation.k_cross_DCF(data, LTR, k, logisticRegression.Quadratic_LR_logLikelihoodRatios, prior , cost_fn, cost_fp, [lam, pi_T])
            print("[5-Folds]  -  lam = 10^-7, pi_T = pi_emp_T: ",min_DCF_LR)
            
            print()

    
    #------------------------RAW FEATURES -----------------
    print("*** minDCF - RAW FEATURES ***")
    Quad_LR_minDCF(Z_normalization(DTR))

    #--------------- GAUSSIANIZED FEATURES-------------------------
    gaussianizedFeatures = gaussianization(DTR)

    print("*** minDCF - GAUSSIANIZED FEATURES  ***")
    Quad_LR_minDCF(gaussianizedFeatures)



def print_graphs_SVM_Cs(DTR, LTR, k ):

    def oneGraphSingleFold(data, prior, cost_fn, cost_fp, pi_T):
        print("working on single fold prior = ", prior)
        exps = numpy.linspace(-3,1, 5)
        Cs = 10** exps
        minDCFs = 0 * exps
        for i in range (Cs.size):
            C= Cs[i]
            minDCFs[i],_ = model_evaluation.singleFold_DCF(data, LTR, SVMClassifier.SVM_computeLogLikelihoods, prior , cost_fn, cost_fp, [pi_T, C])
        
        lb = "minDCF (prior="+ str(prior) +")"
        plt.plot(Cs, minDCFs, label=lb)
        plt.legend()
        print("DONE")

    def oneGraphKFold(data, prior, cost_fn, cost_fp, pi_T):
        print("working on k fold prior = ", prior)
        exps = numpy.linspace(-3,1, 5)
        Cs = 10** exps
        minDCFs = 0 * exps
        for i in range (Cs.size):
            C= Cs[i]
            minDCFs[i],_,_ = model_evaluation.k_cross_DCF(data, LTR,k, SVMClassifier.SVM_computeLogLikelihoods, prior , cost_fn, cost_fp, [pi_T, C])
        
        lb = "minDCF (prior="+ str(prior) +")"
        plt.plot(Cs, minDCFs, label=lb)
        plt.legend()
        print("DONE")

    normalizedFeatures = Z_normalization(DTR)
    gaussianizedFeatures = gaussianization(DTR)

    plt.figure()
    print("1 grafico")
    plt.title("Raw features, single fold")
    plt.xscale('log')
    plt.xlabel("C")
    plt.ylabel("minDCFs")
    oneGraphSingleFold(normalizedFeatures, prior=0.5, cost_fn=1, cost_fp=1, pi_T=0.5)
    oneGraphSingleFold(normalizedFeatures, prior=0.1, cost_fn=1, cost_fp=1, pi_T=0.5)
    oneGraphSingleFold(normalizedFeatures, prior=0.9, cost_fn=1, cost_fp=1, pi_T=0.5)
    plt.savefig('Graph/SVM/linear/singleFoldRAW.png' )
    
    print("2 grafico")
    plt.figure()
    plt.title("Gaussianized features, single fold")
    plt.xscale('log')
    plt.xlabel("C")
    plt.ylabel("minDCFs")
    oneGraphSingleFold(gaussianizedFeatures, prior=0.5, cost_fn=1, cost_fp=1, pi_T=0.5)
    oneGraphSingleFold(gaussianizedFeatures, prior=0.1, cost_fn=1, cost_fp=1, pi_T=0.5)
    oneGraphSingleFold(gaussianizedFeatures, prior=0.9, cost_fn=1, cost_fp=1, pi_T=0.5)
    plt.savefig('Graph/SVM/linear/singleFoldGauss.png' )
   
    print("3 grafico")
    plt.figure()
    plt.title("Raw features, 5 fold")
    plt.xscale('log')
    plt.xlabel("C")
    plt.ylabel("minDCFs")
    oneGraphKFold(normalizedFeatures, prior=0.5, cost_fn=1, cost_fp=1, pi_T=0.5)
    oneGraphKFold(normalizedFeatures, prior=0.1, cost_fn=1, cost_fp=1, pi_T=0.5)
    oneGraphKFold(normalizedFeatures, prior=0.9, cost_fn=1, cost_fp=1, pi_T=0.5)
    plt.savefig('Graph/SVM/linear/5FoldRAW.png' )

    print("4 grafico")
    plt.figure()
    plt.title("Gaussianized features, 5 fold")
    plt.xscale('log')
    plt.xlabel("C")
    plt.ylabel("minDCFs")
    oneGraphKFold(gaussianizedFeatures, prior=0.5, cost_fn=1, cost_fp=1, pi_T=0.5)
    oneGraphKFold(gaussianizedFeatures, prior=0.1, cost_fn=1, cost_fp=1, pi_T=0.5)
    oneGraphKFold(gaussianizedFeatures, prior=0.9, cost_fn=1, cost_fp=1, pi_T=0.5)
    plt.savefig('Graph/SVM/linear/5FoldGauss.png' )
    #plt.show()

def print_table_SVM_minDCF(DTR, LTR, prior, cost_fn, cost_fp, k):

    def linear_SVM_minDCF(data):
            
            C = 0.1
            pi_T = 0.5
            minDCF,_,_ = model_evaluation.k_cross_DCF(data, LTR,k, SVMClassifier.SVM_computeLogLikelihoods, prior , cost_fn, cost_fp, [pi_T, C])
            print("[5-Folds]  -  C= 0.1, pi_T=0.5: ",minDCF)  

            C = 0.1
            pi_T = 0.1
            minDCF,_,_ = model_evaluation.k_cross_DCF(data, LTR,k, SVMClassifier.SVM_computeLogLikelihoods, prior , cost_fn, cost_fp, [pi_T, C])
            print("[5-Folds]  -  C= 0.1, pi_T=0.1: ",minDCF)

            C = 0.1
            pi_T = 0.9
            minDCF,_,_ = model_evaluation.k_cross_DCF(data, LTR,k, SVMClassifier.SVM_computeLogLikelihoods, prior , cost_fn, cost_fp, [pi_T, C])
            print("[5-Folds]  -  C= 0.1, pi_T=0.9: ",minDCF)

            #unbalanced application
            C = 0.1
            pi_T = -1
            minDCF,_,_ = model_evaluation.k_cross_DCF(data, LTR,k, SVMClassifier.SVM_computeLogLikelihoods, prior , cost_fn, cost_fp, [pi_T, C])
            print("[5-Folds]  -  C= 0.1, pi_T=pi_emp_T: ",minDCF)

            print()

    
    #------------------------RAW FEATURES -----------------
    print("*** minDCF - RAW FEATURES ***")
    linear_SVM_minDCF(Z_normalization(DTR))

    #--------------- GAUSSIANIZED FEATURES-------------------------
    gaussianizedFeatures = gaussianization(DTR)

    print("*** minDCF - GAUSSIANIZED FEATURES  ***")
    linear_SVM_minDCF(gaussianizedFeatures)

def print_graphs_Polinomial_SVM_Cs_k_c(DTR, LTR, k ):

    def oneGraphSingleFold(data, prior, cost_fn, cost_fp, pi_T, K, c):
        print("working on single fold k = ", K, "c = ", c)
        exps = numpy.linspace(-2,2, 5)
        Cs = 10** exps
        minDCFs = 0 * exps
        for i in range (Cs.size):
            C= Cs[i]
            minDCFs[i],_,_ = model_evaluation.singleFold_DCF(data, LTR, SVMClassifier.Polinomial_SVM_computeLogLikelihoods, prior , cost_fn, cost_fp, [pi_T, C, c, K])
        
        lb = "minDCF (k="+ str(K) +" c= "+ str(c)+ ")"
        plt.plot(Cs, minDCFs, label=lb)
        plt.legend()
        print("DONE")

    def oneGraphKFold(data, prior, cost_fn, cost_fp, pi_T, K, c):
        print("working on k fold k = ", K, "c = ", c)
        exps = numpy.linspace(-2,2, 5)
        Cs = 10** exps
        minDCFs = 0 * exps
        for i in range (Cs.size):
            C= Cs[i]
            minDCFs[i],_,_ = model_evaluation.k_cross_DCF(data, LTR,k, SVMClassifier.Polinomial_SVM_computeLogLikelihoods, prior , cost_fn, cost_fp, [pi_T, C, c, K])
        
        lb = "minDCF (k="+ str(K) +" c= "+ str(c)+ ")"
        plt.plot(Cs, minDCFs, label=lb)
        plt.legend()
        print("DONE")

    normalizedFeatures = Z_normalization(DTR)
    gaussianizedFeatures = gaussianization(DTR)

    plt.figure()
    print("1 grafico")
    plt.title("Raw features, single fold")
    plt.xscale('log')
    plt.xlabel("C")
    plt.ylabel("minDCFs")
    oneGraphSingleFold(normalizedFeatures, prior=0.5, cost_fn=1, cost_fp=1, pi_T=0.5, K=0.0, c=0.0)
    oneGraphSingleFold(normalizedFeatures, prior=0.5, cost_fn=1, cost_fp=1, pi_T=0.5, K=0.0, c=1.0)
    oneGraphSingleFold(normalizedFeatures, prior=0.5, cost_fn=1, cost_fp=1, pi_T=0.5, K=1.0, c=0.0)
    oneGraphSingleFold(normalizedFeatures, prior=0.5, cost_fn=1, cost_fp=1, pi_T=0.5, K=1.0, c=1.0)

    plt.savefig('Graph/SVM/Quadratic/kc/singleFoldRAW_kc.png' )
    
    plt.figure()
    print("2 grafico")
    plt.title("Gaussianized features, single fold")
    plt.xscale('log')
    plt.xlabel("C")
    plt.ylabel("minDCFs")
    oneGraphSingleFold(gaussianizedFeatures, prior=0.5, cost_fn=1, cost_fp=1, pi_T=0.5, K=0.0, c=0.0)
    oneGraphSingleFold(gaussianizedFeatures, prior=0.5, cost_fn=1, cost_fp=1, pi_T=0.5, K=0.0, c=1.0)
    oneGraphSingleFold(gaussianizedFeatures, prior=0.5, cost_fn=1, cost_fp=1, pi_T=0.5, K=1.0, c=0.0)
    oneGraphSingleFold(gaussianizedFeatures, prior=0.5, cost_fn=1, cost_fp=1, pi_T=0.5, K=1.0, c=1.0)

    plt.savefig('Graph/SVM/Quadratic/kc/singleFoldGAU_kc.png' )
    
    plt.figure()
    print("3 grafico")
    plt.title("Raw features, 5 fold")
    plt.xscale('log')
    plt.xlabel("C")
    plt.ylabel("minDCFs")
    oneGraphKFold(normalizedFeatures, prior=0.5, cost_fn=1, cost_fp=1, pi_T=0.5, K=0.0, c=0.0)
    oneGraphKFold(normalizedFeatures, prior=0.5, cost_fn=1, cost_fp=1, pi_T=0.5, K=0.0, c=1.0)
    oneGraphKFold(normalizedFeatures, prior=0.5, cost_fn=1, cost_fp=1, pi_T=0.5, K=1.0, c=0.0)
    oneGraphKFold(normalizedFeatures, prior=0.5, cost_fn=1, cost_fp=1, pi_T=0.5, K=1.0, c=1.0)

    plt.savefig('Graph/SVM/Quadratic/kc/5FoldRAW_kc.png' )
    
    plt.figure()
    print("4 grafico")
    plt.title("Gaussianized features, 5 fold")
    plt.xscale('log')
    plt.xlabel("C")
    plt.ylabel("minDCFs")
    oneGraphKFold(gaussianizedFeatures, prior=0.5, cost_fn=1, cost_fp=1, pi_T=0.5, K=0.0, c=0.0)
    oneGraphKFold(gaussianizedFeatures, prior=0.5, cost_fn=1, cost_fp=1, pi_T=0.5, K=0.0, c=1.0)
    oneGraphKFold(gaussianizedFeatures, prior=0.5, cost_fn=1, cost_fp=1, pi_T=0.5, K=1.0, c=0.0)
    oneGraphKFold(gaussianizedFeatures, prior=0.5, cost_fn=1, cost_fp=1, pi_T=0.5, K=1.0, c=1.0)

    plt.savefig('Graph/SVM/Quadratic/kc/5FoldGAU_kc.png' )
    
    
 
    

def print_table_Quadratic_SVM_minDCF(DTR, LTR, prior, cost_fn, cost_fp, k): #TODO

    def quadratic_SVM_minDCF(data, C, c, K):
        
        pi_T = 0.5
        minDCF,_,_ = model_evaluation.singleFold_DCF(data, LTR, SVMClassifier.Polinomial_SVM_computeLogLikelihoods, prior , cost_fn, cost_fp, [pi_T, C, c, K])
        print("[5-Folds]  -  C= ", C, ", pi_T=0.5, c= ", c, " k = ",K ,"  : ",minDCF)  

        
        pi_T = 0.1
        minDCF,_,_ = model_evaluation.k_cross_DCF(data, LTR,k, SVMClassifier.SVM_computeLogLikelihoods, prior , cost_fn, cost_fp, [pi_T, C, c, K])
        print("[5-Folds]  -  C= ", C, ", pi_T=0.1, c= ", c, " k = ",K ,"  : ",minDCF)

        
        pi_T = 0.9
        minDCF,_,_ = model_evaluation.k_cross_DCF(data, LTR,k, SVMClassifier.SVM_computeLogLikelihoods, prior , cost_fn, cost_fp, [pi_T, C, c, K])
        print("[5-Folds]  -  C= ", C, ", pi_T=0.9, c= ", c, " k = ",K, "  : ",minDCF)


        
        N = LTR.size #tot number of samples
        n_T = (1*(LTR==1)).sum() #num of samples belonging to the true class
        n_F = (1*(LTR==0)).sum() #num of samples belonging to the false class
        pi_emp_T = n_T / N

        pi_T = pi_emp_T
        minDCF,_,_ = model_evaluation.k_cross_DCF(data, LTR,k, SVMClassifier.SVM_computeLogLikelihoods, prior , cost_fn, cost_fp,[pi_T, C, c, K])
        print("[5-Folds]  - C= ", C, ", pi_T=pi_emp_T, c= ", c, " k = ",K , "  : ",minDCF)


        print()

    def fun_parametri(C,c, K):
        gaussianizedFeatures = gaussianization(DTR)
        normalizedFeatures = Z_normalization(DTR)
        
        
        print("PARAMETRI: (C = " + str(C) + " c= "+ str(c) + "K= " + str(K)+ ")") 

        #------------------------RAW FEATURES -----------------
        print("*** minDCF - RAW FEATURES ***")
        quadratic_SVM_minDCF(normalizedFeatures, C=C, c=c, K=K)

        #--------------- GAUSSIANIZED FEATURES-------------------------
        print("*** minDCF - GAUSSIANIZED FEATURES  ***")
        quadratic_SVM_minDCF(gaussianizedFeatures ,C=C, c=c, K=K)


        print("************************************************")
    
    fun_parametri(10,1,0)
    fun_parametri(100,1,0)
    fun_parametri(0.1,1,1)

def print_graphs_RBF_SVM_Cs(DTR, LTR, k):

    def oneGraphKFold(data, prior, cost_fn, cost_fp, pi_T, loglam):
        print("working on k fold loglam = ", loglam)
        exps = numpy.linspace(-1,2, 10)
        Cs = 10** exps
        minDCFs = 0 * exps
        for i in range (Cs.size):
            C= Cs[i]
            minDCFs[i],_,_ = model_evaluation.k_cross_DCF(data, LTR,k, SVMClassifier.RBF_SVM_computeLogLikelihoods, prior , cost_fn, cost_fp, [pi_T, C, 10**loglam])
        
        lb = " (log(lam)="+ str(loglam) +")"
        plt.plot(Cs, minDCFs, label=lb)
        plt.legend()
        print("DONE")

    normalizedFeatures = Z_normalization(DTR)
    gaussianizedFeatures = gaussianization(DTR)

    print("1 grafico")
    plt.figure()
    plt.title("Raw features, 5 fold")
    plt.xscale('log')
    plt.xlabel("C")
    plt.ylabel("minDCFs")
    oneGraphKFold(normalizedFeatures, prior=0.5, cost_fn=1, cost_fp=1, pi_T=0.5, loglam = 0)
    oneGraphKFold(normalizedFeatures, prior=0.5, cost_fn=1, cost_fp=1, pi_T=0.5, loglam = 0.5)
    oneGraphKFold(normalizedFeatures, prior=0.5, cost_fn=1, cost_fp=1, pi_T=0.5, loglam = 0.8)
    oneGraphKFold(normalizedFeatures, prior=0.5, cost_fn=1, cost_fp=1, pi_T=0.5, loglam = 1)
    plt.savefig('Graph/SVM/RBF/5FoldRAW.png' )

    print("2 grafico")
    plt.figure()
    plt.title("Gaussianized features, 5 fold")
    plt.xscale('log')
    plt.xlabel("C")
    plt.ylabel("minDCFs")
    oneGraphKFold(gaussianizedFeatures, prior=0.5, cost_fn=1, cost_fp=1, pi_T=0.5, loglam = 0)
    oneGraphKFold(gaussianizedFeatures, prior=0.5, cost_fn=1, cost_fp=1, pi_T=0.5, loglam = 0.5)
    oneGraphKFold(gaussianizedFeatures, prior=0.5, cost_fn=1, cost_fp=1, pi_T=0.5, loglam = 0.8)
    oneGraphKFold(gaussianizedFeatures, prior=0.5, cost_fn=1, cost_fp=1, pi_T=0.5, loglam = 1)
    plt.savefig('Graph/SVM/RBF/5FoldGauss.png' )
    #plt.show()

def print_table_RBF_SVM_minDCF(DTR, LTR, prior, cost_fn, cost_fp, k): #TODO


    def RBF_SVM_minDCF(data, C, loglam):
        
        pi_T = 0.5
        minDCF,_,_ = model_evaluation.k_cross_DCF(data, LTR,k, SVMClassifier.RBF_SVM_computeLogLikelihoods, prior , cost_fn, cost_fp, [pi_T, C, 10**loglam])
        print("[5-Folds]  -  C= ", C, ", loglam= ", loglam, " pi_T=0.5: ",minDCF)   

        
        pi_T = 0.1
        minDCF,_,_ = model_evaluation.k_cross_DCF(data, LTR,k, SVMClassifier.RBF_SVM_computeLogLikelihoods, prior , cost_fn, cost_fp, [pi_T, C, 10**loglam])
        print("[5-Folds]  -  C= ", C, ", loglam= ", loglam, " pi_T=0.1: ",minDCF)   

        
        pi_T = 0.9
        minDCF,_,_ = model_evaluation.k_cross_DCF(data, LTR,k, SVMClassifier.RBF_SVM_computeLogLikelihoods, prior , cost_fn, cost_fp, [pi_T, C, 10**loglam])
        print("[5-Folds]  -  C= ", C, ", loglam= ", loglam, " pi_T=0.9: ",minDCF)   


        
        N = LTR.size #tot number of samples
        n_T = (1*(LTR==1)).sum() #num of samples belonging to the true class
        n_F = (1*(LTR==0)).sum() #num of samples belonging to the false class
        pi_emp_T = n_T / N

        pi_T = pi_emp_T
        minDCF,_,_ = model_evaluation.k_cross_DCF(data, LTR,k, SVMClassifier.RBF_SVM_computeLogLikelihoods, prior , cost_fn, cost_fp, [pi_T, C, 10**loglam])
        print("[5-Folds]  -  C= ", C, ", loglam= ", loglam, " pi_T=pi_emp_T: ",minDCF)   


        print()

    def fun_parametri(C,loglam):
        gaussianizedFeatures = gaussianization(DTR)
        normalizedFeatures = Z_normalization(DTR)
        
        
        print("PARAMETRI: (C = " + str(C) + " loglam= "+ str(loglam)+ ")") 

        #------------------------RAW FEATURES -----------------
        print("*** minDCF - RAW FEATURES ***")
        RBF_SVM_minDCF(normalizedFeatures, C=C, loglam=loglam)

        #--------------- GAUSSIANIZED FEATURES-------------------------
        print("*** minDCF - GAUSSIANIZED FEATURES  ***")
        RBF_SVM_minDCF(gaussianizedFeatures ,C=C, loglam=loglam)


        print("************************************************")
    
    fun_parametri(1,0)
    fun_parametri(0.5,0)

    

      
       
def print_graphs_GMM_minDCF(DTR, LTR, k):

    def bar_plot_gmm(raw_minDCFs, gau_minDCFs, gmm_comp, title):

        widthbar = 0.2

        x_ind = numpy.arange(len(gmm_comp))

        raw_ind = x_ind - widthbar/2
        gau_ind = x_ind + widthbar/2

        lb1 = "minDCF (prior=0.5) - Raw"
        lb2 = "minDCF (prior=0.5) - Gaussianized"
        
        plt.figure()
        plt.bar(raw_ind, raw_minDCFs, width = widthbar, color = 'orange', label = lb1)
        plt.bar(gau_ind, gau_minDCFs, width = widthbar, color = 'red', label = lb2)
        plt.title(title)
        plt.xticks(x_ind ,gmm_comp)
        plt.ylabel('minDCFs')
        plt.xlabel('GMM components')
        plt.legend()

        plt.savefig('Graph/GMM/'+title+'.png' )


    def GMM_compute_DCFs(DTR, LTR, k, covariance_type, prior, cost_fn, cost_fp):
        gmm_comp = [1,2,4,8]

        raw_minDCFs = []
        gau_minDCFs = []

        normalized_features = Z_normalization(DTR)
        gaussianizedFeatures = gaussianization(DTR)

        constrained=True
        psi=1
        alpha=0.0001
        delta_l=10**(-6)
    
        print("************************" + covariance_type + "*************************")
        for i in range(len(gmm_comp)):
            params = [constrained, psi, covariance_type, alpha, gmm_comp[i],delta_l]
            print("-------> working on raw data, comp= ", gmm_comp[i])
            # Raw features
            raw_minDCFs_i,_,_ = model_evaluation.k_cross_DCF(normalized_features, LTR, k, gaussian_mixture_models.GMM_computeLogLikelihoodRatios, prior , cost_fn, cost_fp, params)
            print("-------> DONE raw data")
            # Gaussianized features
            print("-------> working on gauss data, comp= ", gmm_comp[i])
            gau_minDCFs_i,_,_ = model_evaluation.k_cross_DCF(gaussianizedFeatures, LTR,k, gaussian_mixture_models.GMM_computeLogLikelihoodRatios, prior , cost_fn, cost_fp, params)
            print("-------> DONE gauss data")
            raw_minDCFs.append(raw_minDCFs_i)
            gau_minDCFs.append(gau_minDCFs_i)    
        
        raw_minDCFs=numpy.array(raw_minDCFs)
        gau_minDCFs=numpy.array(gau_minDCFs)
        return raw_minDCFs, gau_minDCFs, gmm_comp

    
    #### Full Cov
    covariance_type = "Full"
    raw_minDCFs, gau_minDCFs, gmm_comp = GMM_compute_DCFs(DTR, LTR, k, covariance_type, 0.5, 1, 1)
    bar_plot_gmm(raw_minDCFs, gau_minDCFs, gmm_comp, "GMM_Full_covariance")

    #### Diagonal Cov
    covariance_type = "Diagonal"
    raw_minDCFs, gau_minDCFs, gmm_comp = GMM_compute_DCFs(DTR, LTR, k, covariance_type, 0.5, 1, 1)
    bar_plot_gmm(raw_minDCFs, gau_minDCFs, gmm_comp, "GMM_Diagonal_covariance")

    #### Diagonal Cov
    covariance_type = "Tied"
    raw_minDCFs, gau_minDCFs, gmm_comp = GMM_compute_DCFs(DTR, LTR, k, covariance_type, 0.5, 1, 1)
    bar_plot_gmm(raw_minDCFs, gau_minDCFs, gmm_comp, "GMM_Tied_covariance")
    
    #### Diagonal Cov
    covariance_type = "Tied Diagonal"
    raw_minDCFs, gau_minDCFs, gmm_comp = GMM_compute_DCFs(DTR, LTR, k, covariance_type, 0.5, 1, 1)
    bar_plot_gmm(raw_minDCFs, gau_minDCFs, gmm_comp, "GMM_Tied_Diagonal_covariance")

    
    

        

    



def print_table_comparison_DCFs(DTR, LTR, k):

    def actDCF_minDCF(data, llr_calculator, params):
            prior=0.5
            cost_fn=1
            cost_fp=1
            min_DCF_LR,act_DCF_LR,_ = model_evaluation.k_cross_DCF(data, LTR, k, llr_calculator, prior , cost_fn, cost_fp, params)
            print("[5-Folds]  -  prior= 0.5  minDCF: ",min_DCF_LR, " actDCF= ",act_DCF_LR)  

            prior=0.1
            cost_fn=1
            cost_fp=1
            min_DCF_LR,act_DCF_LR,_ = model_evaluation.k_cross_DCF(data, LTR, k, llr_calculator, prior , cost_fn, cost_fp, params)
            print("[5-Folds]  -  prior= 0.1  minDCF: ",min_DCF_LR, " actDCF= ",act_DCF_LR) 

            prior=0.9
            cost_fn=1
            cost_fp=1
            min_DCF_LR,act_DCF_LR,_,_ = model_evaluation.k_cross_DCF(data, LTR, k, llr_calculator, prior , cost_fn, cost_fp, params)
            print("[5-Folds]  -  prior= 0.9  minDCF: ",min_DCF_LR, " actDCF= ",act_DCF_LR) 

            print()

    
    #------------------------FIRST MODEL ----------------- # TODO
    print("*** QuadLog reg, lambda=10**-7, pi_T =0.10.269 Raw features ***")
    lam = 10**(-7)
    pi_T = 0.1
    actDCF_minDCF(Z_normalization(DTR), logisticRegression.Quadratic_LR_logLikelihoodRatios,[lam, pi_T] )

    #--------------- SECOND MODEL-------------------------
    gaussianizedFeatures = gaussianization(DTR)
    print("*** MVG full, gaussianized, noPCA ***")
    actDCF_minDCF(gaussianizedFeatures, MVGclassifiers.MVG_logLikelihoodRatios,[])

def print_err_bayes_plots(data, L, k, llr_calculators, other_params, titles, colors):
    plt.figure()
    plt.title("Bayes Error Plot")
    plt.xlabel("prior log odds")
    plt.ylabel("DCF")
    for i in range (len(llr_calculators)):
        print("Working on calculator "+ str(i))
        model_evaluation.bayes_error_plot(data[i], L, k, llr_calculators[i], other_params[i], titles[i], colors[i] )
        print("DONE")
    plt.savefig('Graph/Error_Bayes_Plots/EBP1.png' )


def compute_DCF_with_optimal_treshold(D, L, k, llr_calculator, otherParams, prior, cost_fn, cost_fp ):
                
    #1st: calculate the loglikelihood ratios using k-cross method
    llr, labels = model_evaluation.k_cross_loglikelihoods(D, L, k, llr_calculator, otherParams)
    
    #2nd: shuffle the loglikelihood ratios = scores
    num_scores = llr.size
    perm = permutation(num_scores)
    llr = llr[perm]
    labels = labels[perm]

    llr1 = llr[:int(num_scores/2)]
    llr2 = llr[int(num_scores/2):]

    labels1 = labels[: int(num_scores/2)]
    labels2 = labels[int(num_scores/2):]

    #minDCF 
    minDCF,_,_,optimal_treshold = model_evaluation.compute_minimum_detection_cost(llr1, labels1, prior , cost_fn, cost_fp)

    predicted_labels = 1*(llr2 > optimal_treshold)

    conf_matrix=model_evaluation.compute_confusion_matrix(predicted_labels, labels2, numpy.unique(labels2).size )
    br= model_evaluation.compute_bayes_risk(conf_matrix, prior, cost_fn, cost_fp)

    #nbr is the DCF obtained with the estimated optimal treshold
    nbr= model_evaluation.compute_normalized_bayes_risk(br, prior, cost_fn, cost_fp)
    
    #actual DCF done with theoretical optimal treshold
    actDCF = model_evaluation.compute_actual_DCF(llr2, labels2, prior , cost_fn, cost_fp)

    #minDCF 
    minDCF,_,_,_ = model_evaluation.compute_minimum_detection_cost(llr2, labels2, prior , cost_fn, cost_fp)


    return (minDCF, actDCF, nbr, optimal_treshold) 


def print_treshold_estimated_table(data, LTR, prior, cost_fn, cost_fp, k, llr_calculator, otherParams, title):
    
    minDCF, actDCF_th, actDCF_opt, optimal_treshold = compute_DCF_with_optimal_treshold(data, LTR, k, llr_calculator, otherParams, prior, cost_fn, cost_fp )

    print(title + ":")
    print("minDCF = ", minDCF)
    print("actual theoretical DCF = ", actDCF_th)
    print("actual optimal DCF = ", actDCF_opt)
    
    return optimal_treshold




def GMM_choosing_hyperparams(DTR, LTR, k, covariance_type, prior, cost_fn, cost_fp):
    gmm_comp = 1
    normalized_features = Z_normalization(DTR)
    gaussianizedFeatures = gaussianization(DTR)

    constrained=True
    #psi_s=[0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]
    #alpha_s=[0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000]
    #delta_l_s=[10**(-6), 10**(-7)]

    psi_s = [0.01]
    alpha_s = [0.1]
    delta_l_s = [10**-6]


    optimal_result = []
    
    print("************************" + covariance_type + "*************************")
    for psi in psi_s:
        for alpha in alpha_s:
            for delta_l in delta_l_s:
    
                params = [constrained, psi, covariance_type, alpha, gmm_comp ,delta_l]
                # Raw features
                raw_minDCFs_i,_,_ = model_evaluation.k_cross_DCF(normalized_features, LTR, k, gaussian_mixture_models.GMM_computeLogLikelihoodRatios, prior , cost_fn, cost_fp, params)
                # Gaussianized features
                gau_minDCFs_i,_,_ = model_evaluation.k_cross_DCF(gaussianizedFeatures, LTR,k, gaussian_mixture_models.GMM_computeLogLikelihoodRatios, prior , cost_fn, cost_fp, params)
                
                print("psi: " + str(psi) + ", alpha: " + str(alpha) + ", delta_l: " + str(delta_l) + " -----> " + "RAW: " + str(raw_minDCFs_i))
                print("psi: " + str(psi) + ", alpha: " + str(alpha) + ", delta_l: " + str(delta_l) + " -----> " + "GAU: " + str(gau_minDCFs_i))

                if(raw_minDCFs_i < 0.4 or gau_minDCFs_i < 0.4):
                    optimal_result.append([raw_minDCFs_i, gau_minDCFs_i])
    return optimal_result
                



if __name__ == '__main__':

    DTR, LTR = load('Data/wine/Train.txt')
    DTE, LTE = load('Data/wine/Test.txt')

    ## DTR: Training Data  
    ## DTE: Evaluation Data
    ## LTR: Training Labels
    ## LTE: Evaluation Labels
    

    ## - compute statistics to analyse the data and the given features
    '''
    # plot histograms of the raw training dataset
    stats.plot_hist(DTR, LTR, "Stat/Hist/Raw")
    
    # plot histograms of the Z_normalized training dataset
    stats.plot_hist(Z_normalization(DTR), LTR, "Stat/Hist/Normalized")
    
    ## - gaussianize the features

    gaussianizedFeatures = gaussianization(DTR)
    stats.plot_hist(gaussianizedFeatures, LTR, "Stat/Hist/Gaussianized")
    
    '''
    '''
    ## heat maps of the gaussianized features to show correlations between features
    stats.plot_heatmaps(gaussianization(DTR), LTR, "Stat/HeatMaps/Gaussianized")
    stats.plot_heatmaps(Z_normalization(DTR), LTR, "Stat/HeatMaps/Normalized")
    '''

    ##enstablish if data are balanced
    '''
    n_high_qty = numpy.count_nonzero(LTR == 1)
    n_low_qty = numpy.count_nonzero(LTR == 0)
    ##-----> number of low qty >> number of high qty
    stats.bars_numsamples(n_high_qty, n_low_qty)
    '''
    
    '''
    NON SERVE
    ## for the balanced application:
    prior = 0.5
    cost_fn = 1
    cost_fp = 1
    classes_prior_probabilties = numpy.array([prior, 1-prior])
    '''
    
    ##choose k for k cross validation
    k = 5

    ##EVAULATION OF THE CLASSIFIERS : 
    ### -- MVG CLASSIFIERS
    '''
    print("********************* MVG TABLE ************************************")
    print("------> pi = 0.5")
    print_table_MVG_classifiers_minDCF(DTR, prior=0.5, cost_fn=1, cost_fp=1, k=k)
    print()

    print("------> pi = 0.9")
    print_table_MVG_classifiers_minDCF(DTR, prior=0.9, cost_fn=1, cost_fp=1, k=k)
    print()

    print("------> pi = 0.1")
    print_table_MVG_classifiers_minDCF(DTR, prior=0.1, cost_fn=1, cost_fp=1, k=k)
    print()
    print("********************************************************************")
    '''
    


    ### -- LOGISTIC REGRESSION
    '''
    print("********************* LR GRAPHS MIN DCF ************************************")
    print_graphs_LR_lambdas(DTR,LTR, k=k)
    print("********************************************************************")
    
    '''
    '''
    
    print("********************* LR TABLE ************************************")
    print("------> applicazione prior = 0.5")
    print_table_LR_minDCF(DTR,LTR, prior=0.5, cost_fn=1, cost_fp=1, k=k)
    print()
    print("------> applicazione con prior = 0.1")
    print_table_LR_minDCF(DTR,LTR, prior=0.1, cost_fn=1, cost_fp=1, k=k)
    print()
    print("------> applicazione con prior = 0.9")
    print_table_LR_minDCF(DTR, LTR, prior=0.9, cost_fn=1, cost_fp=1, k=k)
    print()
    print("********************************************************************")
    '''
    '''
    
    print("********************* quadratic LR GRAPHS ************************************")
    print_graphs_quadratic_LR_lambdas(DTR, LTR,  k)
    print("********************************************************************")
    '''
    '''
    print("********************* QUADRATIC LR TABLE ************************************")
    print("------> applicazione prior = 0.5")
    print_table_Quadratic_LR_minDCF(DTR,LTR, prior=0.5, cost_fn=1, cost_fp=1, k=k)
    print()
    print("------> applicazione con prior = 0.1")
    print_table_Quadratic_LR_minDCF(DTR,LTR, prior=0.1, cost_fn=1, cost_fp=1, k=k)
    print()
    print("------> applicazione con prior = 0.9")
    print_table_Quadratic_LR_minDCF(DTR, LTR, prior=0.9, cost_fn=1, cost_fp=1, k=k)
    print()
    print("********************************************************************")
    '''
    ### -- SVM
    
    '''
    print("********************* SVM GRAPHS ************************************")
    print_graphs_SVM_Cs(DTR, LTR, k=k )
    print("********************************************************************")
    '''
    '''
    print("********************* SVM TABLES ************************************")
    print("------> applicazione con prior = 0.5")
    print_table_SVM_minDCF(DTR, LTR, prior=0.5, cost_fn=1, cost_fp=1, k=k )
    print("------> applicazione con prior = 0.9")
    print_table_SVM_minDCF(DTR, LTR, prior=0.9, cost_fn=1, cost_fp=1, k=k )
    print("------> applicazione con prior = 0.1")
    print_table_SVM_minDCF(DTR, LTR, prior=0.1, cost_fn=1, cost_fp=1, k=k )
    print("********************************************************************")
    '''

    '''
    
    print("********************* quadratic SVM GRAPHS ************************************")
    print_graphs_Polinomial_SVM_Cs(DTR, LTR, k=k )
    print("********************************************************************")
    '''
    '''
    print("********************* quadratic SVM GRAPHS changing C,k,c ************************************")
    print_graphs_Polinomial_SVM_Cs_k_c(DTR, LTR, k=k )
    print("********************************************************************")
    '''
    
    '''
    print("********************* quadratic SVM TABLES ************************************")
    print("------> applicazione con prior = 0.5")
    print_table_Quadratic_SVM_minDCF(DTR, LTR, prior=0.5, cost_fn=1, cost_fp=1, k=k )
    print("------> applicazione con prior = 0.9")
    print_table_Quadratic_SVM_minDCF(DTR, LTR, prior=0.9, cost_fn=1, cost_fp=1, k=k )
    print("------> applicazione con prior = 0.1")
    print_table_Quadratic_SVM_minDCF(DTR, LTR, prior=0.1, cost_fn=1, cost_fp=1, k=k )
    print("********************************************************************")
    '''
    
    '''
    print("********************* RBF SVM GRAPHS ************************************")
    print_graphs_RBF_SVM_Cs(DTR, LTR, k=k )
    print("********************************************************************")
    '''

    
    '''
    print("********************* RBF SVM TABLES ************************************")
    print("------> applicazione con prior = 0.5")
    print_table_RBF_SVM_minDCF(DTR, LTR, prior=0.5, cost_fn=1, cost_fp=1, k=k )
    print("------> applicazione con prior = 0.9")
    print_table_RBF_SVM_minDCF(DTR, LTR, prior=0.9, cost_fn=1, cost_fp=1, k=k )
    print("------> applicazione con prior = 0.1")
    print_table_RBF_SVM_minDCF(DTR, LTR, prior=0.1, cost_fn=1, cost_fp=1, k=k )
    print("********************************************************************")
    '''
    
    ### GMM
    
    
    

    '''
    #### Full Cov
    covariance_type = "Full"
    print( GMM_choosing_hyperparams(DTR, LTR, k, covariance_type, 0.5, 1, 1))

    #### Diagonal Cov
    covariance_type = "Diagonal"
    print(GMM_choosing_hyperparams(DTR, LTR, k, covariance_type, 0.5, 1, 1))

    #### Diagonal Cov
    covariance_type = "Tied"
    print(GMM_choosing_hyperparams(DTR, LTR, k, covariance_type, 0.5, 1, 1))
    
    #### Diagonal Cov
    covariance_type = "Tied Diagonal"
    print(GMM_choosing_hyperparams(DTR, LTR, k, covariance_type, 0.5, 1, 1))
    '''
    
    
    print_graphs_GMM_minDCF(DTR, LTR, k)
    

    ## COMPARISON BETWEEN ACT DCF AND MIN DCF OF THE CHOSEN MODELS
    '''
    print_table_comparison_DCFs(DTR, LTR, k=k)
    
    #error bayes plot
    
    lam = 10**(-7)
    pi_T = 0.1
    data = [Z_normalization(DTR), gaussianization(DTR)]
    llr_calculators = [logisticRegression.Quadratic_LR_logLikelihoodRatios,MVGclassifiers.MVG_logLikelihoodRatios ]
    other_params = [[lam, pi_T], []]
    titles = ["Quad Log reg", "MVG Full cov"]
    colors = ["r", "b"]
    print_err_bayes_plots(data, LTR, k, llr_calculators, other_params, titles, colors)
    
    
    lam = 10**(-7)
    pi_T = 0.1

    print("------>Treshold estimated table:")
    print()
    print("------> applicazione prior = 0.5")
    prior = 0.5
    print_treshold_estimated_table(Z_normalization(DTR), LTR, prior, 1, 1, k, logisticRegression.Quadratic_LR_logLikelihoodRatios, [lam, pi_T], "Quad Log Reg")
    print_treshold_estimated_table(gaussianization(DTR), LTR, prior, 1, 1, k, MVGclassifiers.MVG_logLikelihoodRatios, [], "MVG with full cov")
    print()

    print("------> applicazione prior = 0.1")
    prior = 0.1
    print_treshold_estimated_table(Z_normalization(DTR), LTR, prior, 1, 1, k, logisticRegression.Quadratic_LR_logLikelihoodRatios, [lam, pi_T], "Quad Log Reg")
    print_treshold_estimated_table(gaussianization(DTR), LTR, prior, 1, 1, k, MVGclassifiers.MVG_logLikelihoodRatios, [], "MVG with full cov")
    print()


    print("------> applicazione prior = 0.9")
    prior = 0.5
    print_treshold_estimated_table(Z_normalization(DTR), LTR, prior, 1, 1, k, logisticRegression.Quadratic_LR_logLikelihoodRatios, [lam, pi_T], "Quad Log Reg")
    print_treshold_estimated_table(gaussianization(DTR), LTR, prior, 1, 1, k, MVGclassifiers.MVG_logLikelihoodRatios, [], "MVG with full cov")
    print()
    '''