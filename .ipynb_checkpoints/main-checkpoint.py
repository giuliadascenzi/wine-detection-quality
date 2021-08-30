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
            pi_T = 0.5
            min_DCF_LR,_,_ = model_evaluation.k_cross_DCF(data, LTR, k, logisticRegression.Quadratic_LR_logLikelihoodRatios, prior , cost_fn, cost_fp, [lam, pi_T])
            print("[5-Folds]  -  lam = 10^-7, pi_T = 0.5: ",min_DCF_LR)  

            pi_T = 0.1
            min_DCF_LR,_,_ = model_evaluation.k_cross_DCF(data, LTR, k, logisticRegression.Quadratic_LR_logLikelihoodRatios, prior , cost_fn, cost_fp, [lam, pi_T])
            print("[5-Folds]  -  lam = 10^-7, pi_T = 0.1: ",min_DCF_LR)

            pi_T = 0.9
            min_DCF_LR,_,_ = model_evaluation.k_cross_DCF(data, LTR, k, logisticRegression.Quadratic_LR_logLikelihoodRatios, prior , cost_fn, cost_fp, [lam, pi_T])
            print("[5-Folds]  -  lam = 10^-7, pi_T = 0.9: ",min_DCF_LR)

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

    

def print_graphs_Polinomial_SVM_Cs(DTR, LTR, k ):

    def oneGraphSingleFold(data, prior, cost_fn, cost_fp, pi_T):
        print("working on single fold prior = ", prior)
        exps = numpy.linspace(-3,1, 5)
        Cs = 10** exps
        minDCFs = 0 * exps
        for i in range (Cs.size):
            C= Cs[i]
            minDCFs[i],_ = model_evaluation.singleFold_DCF(data, LTR, SVMClassifier.Polinomial_SVM_computeLogLikelihoods, prior , cost_fn, cost_fp, [pi_T, C])
        
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
            minDCFs[i],_,_ = model_evaluation.k_cross_DCF(data, LTR,k, SVMClassifier.Polinomial_SVM_computeLogLikelihoods, prior , cost_fn, cost_fp, [pi_T, C])
        
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
    plt.savefig('Graph/SVM/Quadratic/singleFoldRAW.png' )
    
    print("2 grafico")
    plt.figure()
    plt.title("Gaussianized features, single fold")
    plt.xscale('log')
    plt.xlabel("C")
    plt.ylabel("minDCFs")
    oneGraphSingleFold(gaussianizedFeatures, prior=0.5, cost_fn=1, cost_fp=1, pi_T=0.5)
    oneGraphSingleFold(gaussianizedFeatures, prior=0.1, cost_fn=1, cost_fp=1, pi_T=0.5)
    oneGraphSingleFold(gaussianizedFeatures, prior=0.9, cost_fn=1, cost_fp=1, pi_T=0.5)
    plt.savefig('Graph/SVM/Quadratic/singleFoldGauss.png' )
   
    print("3 grafico")
    plt.figure()
    plt.title("Raw features, 5 fold")
    plt.xscale('log')
    plt.xlabel("C")
    plt.ylabel("minDCFs")
    oneGraphKFold(normalizedFeatures, prior=0.5, cost_fn=1, cost_fp=1, pi_T=0.5)
    oneGraphKFold(normalizedFeatures, prior=0.1, cost_fn=1, cost_fp=1, pi_T=0.5)
    oneGraphKFold(normalizedFeatures, prior=0.9, cost_fn=1, cost_fp=1, pi_T=0.5)
    plt.savefig('Graph/SVM/Quadratic/5FoldRAW.png' )

    print("4 grafico")
    plt.figure()
    plt.title("Gaussianized features, 5 fold")
    plt.xscale('log')
    plt.xlabel("C")
    plt.ylabel("minDCFs")
    oneGraphKFold(gaussianizedFeatures, prior=0.5, cost_fn=1, cost_fp=1, pi_T=0.5)
    oneGraphKFold(gaussianizedFeatures, prior=0.1, cost_fn=1, cost_fp=1, pi_T=0.5)
    oneGraphKFold(gaussianizedFeatures, prior=0.9, cost_fn=1, cost_fp=1, pi_T=0.5)
    plt.savefig('Graph/SVM/Quadratic/5FoldGauss.png' )
    #plt.show()



def print_graphs_RBF_SVM_Cs(DTR, LTR, k):

    def oneGraphKFold(data, prior, cost_fn, cost_fp, pi_T, loglam):
        print("working on k fold loglam = ", loglam)
        exps = numpy.linspace(-1,3, 10)
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
    oneGraphKFold(normalizedFeatures, prior=0.5, cost_fn=1, cost_fp=1, pi_T=0.5, loglam = -1)
    oneGraphKFold(normalizedFeatures, prior=0.5, cost_fn=1, cost_fp=1, pi_T=0.5, loglam = -2)
    oneGraphKFold(normalizedFeatures, prior=0.5, cost_fn=1, cost_fp=1, pi_T=0.5, loglam = -3)
    plt.savefig('Graph/SVM/RBF/5FoldRAW.png' )

    print("2 grafico")
    plt.figure()
    plt.title("Gaussianized features, 5 fold")
    plt.xscale('log')
    plt.xlabel("C")
    plt.ylabel("minDCFs")
    oneGraphKFold(gaussianizedFeatures, prior=0.5, cost_fn=1, cost_fp=1, pi_T=0.5, loglam = 0)
    oneGraphKFold(gaussianizedFeatures, prior=0.5, cost_fn=1, cost_fp=1, pi_T=0.5, loglam = -1)
    oneGraphKFold(gaussianizedFeatures, prior=0.5, cost_fn=1, cost_fp=1, pi_T=0.5, loglam = -2)
    oneGraphKFold(gaussianizedFeatures, prior=0.5, cost_fn=1, cost_fp=1, pi_T=0.5, loglam = -3)
    plt.savefig('Graph/SVM/RBF/5FoldGauss.png' )
    #plt.show()

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
            min_DCF_LR,act_DCF_LR,_ = model_evaluation.k_cross_DCF(data, LTR, k, llr_calculator, prior , cost_fn, cost_fp, params)
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
    plt.figure()
    plt.xlabel("nsamples")
    plt.hist(numpy.zeros(n_low_qty),  label = 'low quality')
    plt.hist(numpy.ones(n_high_qty), label = 'high quality')
    plt.legend()
    plt.savefig('Stat/hist_number_of_data.png')
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

    
    print("********************* quadratic SVM GRAPHS ************************************")
    print_graphs_Polinomial_SVM_Cs(DTR, LTR, k=k )
    print("********************************************************************")

    
    print("********************* RBF SVM GRAPHS ************************************")
    print_graphs_RBF_SVM_Cs(DTR, LTR, k=k )
    print("********************************************************************")
    
    
    '''
    ## COMPARISON BETWEEN ACT DCF AND MIN DCF OF THE CHOSEN MODELS
    '''
    print_table_comparison_DCFs(DTR, LTR, k=k)
    '''
    #error bayes plot
    lam = 10**(-7)
    pi_T = 0.1
    data = [Z_normalization(DTR), gaussianization(DTR)]
    llr_calculators = [logisticRegression.Quadratic_LR_logLikelihoodRatios,MVGclassifiers.MVG_logLikelihoodRatios ]
    other_params = [[lam, pi_T], []]
    titles = ["Quad Log reg", "MVG Full cov"]
    colors = ["r", "b"]
    print_err_bayes_plots(data, LTR, k, llr_calculators, other_params, titles, colors)


    
   

          