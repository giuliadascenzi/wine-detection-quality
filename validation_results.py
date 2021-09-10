import numpy
import scipy.stats
import preprocessing
import model_validation
import MVGclassifiers
import dimensionality_reduction_techniques as redTec
import logisticRegression
import matplotlib.pyplot as plt
import SVMClassifier
import gaussian_mixture_models 


def print_table_MVG_classifiers_minDCF(DTR, LTR, prior, cost_fn, cost_fp, k):

    def MVG_Classifiers_minDCF(data):
        #Full_Cov 
        min_DCF_MVG,_,_ = model_validation.singleFold_DCF(data, LTR, MVGclassifiers.MVG_logLikelihoodRatios, prior , cost_fn, cost_fp)
        print("[Single Fold] -  MVG: ",min_DCF_MVG)  
        min_DCF_MVG,_,_ = model_validation.k_cross_DCF(data, LTR, k, MVGclassifiers.MVG_logLikelihoodRatios, prior , cost_fn, cost_fp)
        print("[5-Folds]  -  MVG: ",min_DCF_MVG)  

        #Diag_Cov == Naive
        min_DCF_Diag_Cov,_,_ = model_validation.singleFold_DCF(data, LTR, MVGclassifiers.NAIVE_logLikelihoodRatios, prior , cost_fn, cost_fp)
        print("[Single Fold]  - MVG with Diag cov: ",min_DCF_Diag_Cov)
        min_DCF_Diag_Cov,_,_ = model_validation.k_cross_DCF(data, LTR,k, MVGclassifiers.NAIVE_logLikelihoodRatios, prior , cost_fn, cost_fp)
        print("[5- Fold] - MVG with Diag cov: ",min_DCF_Diag_Cov)

        #Tied
        min_DCF_Tied,_,_ = model_validation.singleFold_DCF(data, LTR, MVGclassifiers.TIED_logLikelihoodRatios, prior , cost_fn, cost_fp)
        print("[Single Fold] - Tied MVG: ",min_DCF_Tied)
        min_DCF_Tied,_,_ = model_validation.k_cross_DCF(data, LTR,k, MVGclassifiers.TIED_logLikelihoodRatios, prior , cost_fn, cost_fp)
        print("[5- Fold] - Tied MVG: ",min_DCF_Tied)

        #Tied Diag_Cov
        min_DCF_Tied_Diag_Cov,_,_ = model_validation.singleFold_DCF(data, LTR, MVGclassifiers.TIED_DIAG_COV_logLikelihoodRatios, prior , cost_fn, cost_fp)
        print("[Single Fold] - Tied MVG with Diag Cov: ",min_DCF_Tied_Diag_Cov)
        min_DCF_Tied_Diag_Cov,_,_ = model_validation.k_cross_DCF(data, LTR, k,  MVGclassifiers.TIED_DIAG_COV_logLikelihoodRatios, prior , cost_fn, cost_fp)
        print("[5 Fold] - Tied MVG with Diag Cov: ",min_DCF_Tied_Diag_Cov)

        print()

    #!!! normalization is important before PCA
    normalized_data = preprocessing.Z_normalization(DTR)
    
    #------------------------RAW FEATURES (normalized) -----------------
    print("*** minDCF - RAW (normalized) FEATURES - NO PCA ***")
    MVG_Classifiers_minDCF(normalized_data)
    
    #------------------------RAW FEATURES (normalized) WITH PCA = 10 --------------------
    principal_components10 = redTec.PCA(normalized_data, 10)
    print("*** minDCF - RAW (normalized) FEATURES -  PCA (m=10) ***")
    MVG_Classifiers_minDCF(principal_components10)       


    #------------------------RAW FEATURES (normalized) WITH PCA = 9 --------------------
    principal_components9 = redTec.PCA(normalized_data, 9)
    print("*** minDCF - RAW (normalized) FEATURES -  PCA (m=9) ***")
    MVG_Classifiers_minDCF(principal_components9)       


    #------------------------RAW FEATURES (normalized) WITH PCA = 8 --------------------
    
    principal_components8= redTec.PCA(normalized_data, 8)
    print("*** minDCF - RAW (normalized) FEATURES -  PCA (m=8) ***")
    MVG_Classifiers_minDCF(principal_components8)    


    ## Z --> PCA --> GAUSS
    #--------------- GAUSSIANIZED FEATURES-------------------------
    gaussianizedFeatures = preprocessing.gaussianization(DTR)
    print("*** minDCF - GAUSSIANIZED FEATURES - NO PCA ***")
    MVG_Classifiers_minDCF(gaussianizedFeatures)

    #------------------------GAUSSIANIZED FEATURES WITH PCA = 10 --------------------
    print("*** minDCF - GAUSSIANIZED FEATURES -  PCA m=10 ***")
    principal_components10= redTec.PCA(normalized_data, 10)
    gaussianized_principal_components_10 = preprocessing.gaussianization(principal_components10)
    MVG_Classifiers_minDCF(gaussianized_principal_components_10)     

    #------------------------GAUSSIANIZED FEATURES WITH PCA = 9 --------------------
    print("*** minDCF - GAUSSIANIZED FEATURES -  PCA m=9 ***")
    principal_components9= redTec.PCA(normalized_data, 9)
    gaussianized_principal_components_9 = preprocessing.gaussianization(principal_components9)
    MVG_Classifiers_minDCF(gaussianized_principal_components_9)     


    #------------------------GAUSSIANIZED FEATURES WITH PCA = 8 --------------------
    print("*** minDCF - GAUSSIANIZED FEATURES -  PCA m=8 ***")
    principal_components8= redTec.PCA(normalized_data, 8)
    gaussianized_principal_components_8= preprocessing.gaussianization(principal_components8)
    MVG_Classifiers_minDCF(gaussianized_principal_components_8)

#--------------------------

def print_table_LR_minDCF(DTR, LTR, prior, cost_fn, cost_fp, k):

    def LR_minDCF(data):
            
        lam = 10**(-7)
        pi_T = 0.5
        min_DCF_LR,_,_ = model_validation.k_cross_DCF(data, LTR, k, logisticRegression.LR_logLikelihoodRatios, prior , cost_fn, cost_fp, [lam, pi_T])
        print("[5-Folds]  -  lam = 10^-7, pi_T = 0.5: ",min_DCF_LR)  

        pi_T = 0.1
        min_DCF_LR,_,_ = model_validation.k_cross_DCF(data, LTR, k, logisticRegression.LR_logLikelihoodRatios, prior , cost_fn, cost_fp, [lam, pi_T])
        print("[5-Folds]  -  lam = 10^-7, pi_T = 0.1: ",min_DCF_LR)

        pi_T = 0.9
        min_DCF_LR,_,_ = model_validation.k_cross_DCF(data, LTR, k, logisticRegression.LR_logLikelihoodRatios, prior , cost_fn, cost_fp, [lam, pi_T])
        print("[5-Folds]  -  lam = 10^-7, pi_T = 0.9: ",min_DCF_LR)
        

        N = LTR.size #tot number of samples
        n_T = (1*(LTR==1)).sum() #num of samples belonging to the true class
        pi_emp_T = n_T / N

        pi_T = pi_emp_T
        
        min_DCF_LR,_,_ = model_validation.k_cross_DCF(data, LTR, k, logisticRegression.LR_logLikelihoodRatios, prior , cost_fn, cost_fp, [lam, pi_T])
        print("[5-Folds]  -  lam = 10^-7, pi_T = pi_emp_T: ",min_DCF_LR)

        print()

    
    #------------------------RAW FEATURES -----------------
    print("*** minDCF - RAW FEATURES ***")
    LR_minDCF(preprocessing.Z_normalization(DTR))

    #--------------- GAUSSIANIZED FEATURES-------------------------
    gaussianizedFeatures = preprocessing.gaussianization(DTR)

    print("*** minDCF - GAUSSIANIZED FEATURES  ***")
    LR_minDCF(gaussianizedFeatures)

#--------------------------

def print_graphs_LR_lambdas(DTR, LTR,  k):
    def oneGraphSingleFold( data, prior, cost_fn, cost_fp, pi_T):
        print("working on single fold prior = ", prior)
        exps = numpy.linspace(-8,5, 14)
        lambdas = 10** exps
        minDCFs = 0 * exps
        for i in range (lambdas.size):
            lam= lambdas[i]
            minDCFs[i],_,_ = model_validation.singleFold_DCF(data, LTR, logisticRegression.LR_logLikelihoodRatios, prior , cost_fn, cost_fp, [lam, pi_T])
        
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
            minDCFs[i],_,_ = model_validation.k_cross_DCF(data, LTR,k, logisticRegression.LR_logLikelihoodRatios, prior , cost_fn, cost_fp, [lam, pi_T])
        
        lb = "minDCF (prior="+ str(prior)+ " )"
        print("DONE")
        plt.plot(lambdas, minDCFs, label=lb)
        plt.legend()

    normalizedFeatures = preprocessing.Z_normalization(DTR)
    gaussianizedFeatures = preprocessing.gaussianization(DTR)

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

#--------------------------

def print_graphs_quadratic_LR_lambdas(DTR, LTR,  k):
    def oneGraphSingleFold( data, prior, cost_fn, cost_fp, pi_T):
        print("working on single fold prior = ", prior)
        exps = numpy.linspace(-8,5, 14)
        lambdas = 10** exps
        minDCFs = 0 * exps
        for i in range (lambdas.size):
            lam= lambdas[i]
            minDCFs[i],_,_ = model_validation.singleFold_DCF(data, LTR, logisticRegression.Quadratic_LR_logLikelihoodRatios, prior , cost_fn, cost_fp, [lam, pi_T])
        
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
            minDCFs[i],_,_ = model_validation.k_cross_DCF(data, LTR,k, logisticRegression.Quadratic_LR_logLikelihoodRatios, prior , cost_fn, cost_fp, [lam, pi_T])
        
        lb = "minDCF (prior="+ str(prior)+ " )"
        print("DONE")
        plt.plot(lambdas, minDCFs, label=lb)
        plt.legend()

    normalizedFeatures = preprocessing.Z_normalization(DTR)
    gaussianizedFeatures = preprocessing.gaussianization(DTR)

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

#--------------------------

def print_table_Quadratic_LR_minDCF(DTR, LTR, prior, cost_fn, cost_fp, k):

    def Quad_LR_minDCF(data):
            lam = 10**(-2)
            
            pi_T = 0.5
            min_DCF_LR,_,_ = model_validation.k_cross_DCF(data, LTR, k, logisticRegression.Quadratic_LR_logLikelihoodRatios, prior , cost_fn, cost_fp, [lam, pi_T])
            print("[5-Folds]  -  lam = 10^-2, pi_T = 0.5: ",min_DCF_LR)  

            pi_T = 0.1
            min_DCF_LR,_,_ = model_validation.k_cross_DCF(data, LTR, k, logisticRegression.Quadratic_LR_logLikelihoodRatios, prior , cost_fn, cost_fp, [lam, pi_T])
            print("[5-Folds]  -  lam = 10^-2, pi_T = 0.1: ",min_DCF_LR)

            pi_T = 0.9
            min_DCF_LR,_,_ = model_validation.k_cross_DCF(data, LTR, k, logisticRegression.Quadratic_LR_logLikelihoodRatios, prior , cost_fn, cost_fp, [lam, pi_T])
            print("[5-Folds]  -  lam = 10^-2, pi_T = 0.9: ",min_DCF_LR)
            

            N = LTR.size #tot number of samples
            n_T = (1*(LTR==1)).sum() #num of samples belonging to the true class
            pi_emp_T = n_T / N

            pi_T = pi_emp_T
            min_DCF_LR,_,_ = model_validation.k_cross_DCF(data, LTR, k, logisticRegression.Quadratic_LR_logLikelihoodRatios, prior , cost_fn, cost_fp, [lam, pi_T])
            print("[5-Folds]  -  lam = 10^-2, pi_T = pi_emp_T: ",min_DCF_LR)
            
            print()

    
    #------------------------RAW FEATURES -----------------
    print("*** minDCF - RAW FEATURES ***")
    Quad_LR_minDCF(preprocessing.Z_normalization(DTR))

    #--------------- GAUSSIANIZED FEATURES-------------------------
    gaussianizedFeatures = preprocessing.gaussianization(DTR)

    print("*** minDCF - GAUSSIANIZED FEATURES  ***")
    Quad_LR_minDCF(gaussianizedFeatures)

#--------------------------

def print_graphs_SVM_Cs(DTR, LTR, k ):

    def oneGraphSingleFold(data, prior, cost_fn, cost_fp, pi_T):
        print("working on single fold prior = ", prior)
        exps = numpy.linspace(-3,1, 5)
        Cs = 10** exps
        minDCFs = 0 * exps
        for i in range (Cs.size):
            C= Cs[i]
            minDCFs[i],_,_ = model_validation.singleFold_DCF(data, LTR, SVMClassifier.SVM_computeLogLikelihoods, prior , cost_fn, cost_fp, [pi_T, C])
        
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
            minDCFs[i],_,_ = model_validation.k_cross_DCF(data, LTR,k, SVMClassifier.SVM_computeLogLikelihoods, prior , cost_fn, cost_fp, [pi_T, C])
        
        lb = "minDCF (prior="+ str(prior) +")"
        plt.plot(Cs, minDCFs, label=lb)
        plt.legend()
        print("DONE")

    normalizedFeatures = preprocessing.Z_normalization(DTR)
    gaussianizedFeatures = preprocessing.gaussianization(DTR)

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

#--------------------------

def print_table_SVM_minDCF(DTR, LTR, prior, cost_fn, cost_fp, k):

    def linear_SVM_minDCF(data):
        C = 0.1
        
        pi_T = 0.5
        minDCF,_,_ = model_validation.k_cross_DCF(data, LTR,k, SVMClassifier.SVM_computeLogLikelihoods, prior , cost_fn, cost_fp, [pi_T, C])
        print("[5-Folds]  -  C= 0.1, pi_T=0.5: ",minDCF)  

        
        pi_T = 0.1
        minDCF,_,_ = model_validation.k_cross_DCF(data, LTR,k, SVMClassifier.SVM_computeLogLikelihoods, prior , cost_fn, cost_fp, [pi_T, C])
        print("[5-Folds]  -  C= 0.1, pi_T=0.1: ",minDCF)

        
        pi_T = 0.9
        minDCF,_,_ = model_validation.k_cross_DCF(data, LTR,k, SVMClassifier.SVM_computeLogLikelihoods, prior , cost_fn, cost_fp, [pi_T, C])
        print("[5-Folds]  -  C= 0.1, pi_T=0.9: ",minDCF)

        #unbalanced application
        
        pi_T = -1
        minDCF,_,_ = model_validation.k_cross_DCF(data, LTR,k, SVMClassifier.SVM_computeLogLikelihoods, prior , cost_fn, cost_fp, [pi_T, C])
        print("[5-Folds]  -  C= 0.1, pi_T=pi_emp_T: ",minDCF)

        print()

    
    #------------------------RAW FEATURES -----------------
    print("*** minDCF - RAW FEATURES ***")
    linear_SVM_minDCF(preprocessing.Z_normalization(DTR))

    #--------------- GAUSSIANIZED FEATURES-------------------------
    gaussianizedFeatures = preprocessing.gaussianization(DTR)

    print("*** minDCF - GAUSSIANIZED FEATURES  ***")
    linear_SVM_minDCF(gaussianizedFeatures)

#--------------------------

def print_graphs_Polinomial_SVM_Cs(DTR, LTR, k ):

    def oneGraphSingleFold(data, prior, cost_fn, cost_fp, pi_T, K, c):
        print("working on single fold prior = ",prior)
        exps = numpy.linspace(-2,2, 5)
        Cs = 10** exps
        minDCFs = 0 * exps
        for i in range (Cs.size):
            C= Cs[i]
            minDCFs[i],_,_ = model_validation.singleFold_DCF(data, LTR, SVMClassifier.Polinomial_SVM_computeLogLikelihoods, prior , cost_fn, cost_fp, [pi_T, C, c, K])
        
        lb = "minDCF (prior= ", prior, ")"
        plt.plot(Cs, minDCFs, label=lb)
        plt.legend()
        print("DONE")

    def oneGraphKFold(data, prior, cost_fn, cost_fp, pi_T, K, c):
        print("working on k fold prior = ",prior)
        exps = numpy.linspace(-2,2, 5)
        Cs = 10** exps
        minDCFs = 0 * exps
        for i in range (Cs.size):
            C= Cs[i]
            minDCFs[i],_,_ = model_validation.k_cross_DCF(data, LTR,k, SVMClassifier.Polinomial_SVM_computeLogLikelihoods, prior , cost_fn, cost_fp, [pi_T, C, c, K])
        
        lb = "minDCF (prior= ", prior, ")"
        plt.plot(Cs, minDCFs, label=lb)
        plt.legend()
        print("DONE")

    normalizedFeatures = preprocessing.Z_normalization(DTR)
    gaussianizedFeatures = preprocessing.gaussianization(DTR)

    plt.figure()
    print("1 grafico")
    plt.title("Raw features, single fold")
    plt.xscale('log')
    plt.xlabel("C")
    plt.ylabel("minDCFs")
    oneGraphSingleFold(normalizedFeatures, prior=0.5, cost_fn=1, cost_fp=1, pi_T=0.5, K=0.0, c=1.0)
    oneGraphSingleFold(normalizedFeatures, prior=0.1, cost_fn=1, cost_fp=1, pi_T=0.5, K=0.0, c=1.0)
    oneGraphSingleFold(normalizedFeatures, prior=0.9, cost_fn=1, cost_fp=1, pi_T=0.5,  K=0.0, c=1.0)
    

    plt.savefig('Graph/SVM/Quadratic/singleFoldRAW.png' )
    
    plt.figure()
    print("2 grafico")
    plt.title("Gaussianized features, single fold")
    plt.xscale('log')
    plt.xlabel("C")
    plt.ylabel("minDCFs")
    oneGraphSingleFold(gaussianizedFeatures, prior=0.5, cost_fn=1, cost_fp=1, pi_T=0.5, K=0.0, c=1.0)
    oneGraphSingleFold(gaussianizedFeatures, prior=0.1, cost_fn=1, cost_fp=1, pi_T=0.5, K=0.0, c=1.0)
    oneGraphSingleFold(gaussianizedFeatures, prior=0.9, cost_fn=1, cost_fp=1, pi_T=0.5,  K=0.0, c=1.0)
    

    plt.savefig('Graph/SVM/Quadratic/singleFoldGAU.png' )
    
    plt.figure()
    print("3 grafico")
    plt.title("Raw features, 5 fold")
    plt.xscale('log')
    plt.xlabel("C")
    plt.ylabel("minDCFs")
    oneGraphKFold(normalizedFeatures, prior=0.5, cost_fn=1, cost_fp=1, pi_T=0.5, K=0.0, c=1.0)
    oneGraphKFold(normalizedFeatures, prior=0.1, cost_fn=1, cost_fp=1, pi_T=0.5, K=0.0, c=1.0)
    oneGraphKFold(normalizedFeatures, prior=0.9, cost_fn=1, cost_fp=1, pi_T=0.5,  K=0.0, c=1.0)
    plt.savefig('Graph/SVM/Quadratic/5FoldRAW.png' )
    
    plt.figure()
    print("4 grafico")
    plt.title("Gaussianized features, 5 fold")
    plt.xscale('log')
    plt.xlabel("C")
    plt.ylabel("minDCFs")
    oneGraphKFold(gaussianizedFeatures, prior=0.5, cost_fn=1, cost_fp=1, pi_T=0.5, K=0.0, c=1.0)
    oneGraphKFold(gaussianizedFeatures, prior=0.1, cost_fn=1, cost_fp=1, pi_T=0.5, K=0.0, c=1.0)
    oneGraphKFold(gaussianizedFeatures, prior=0.9, cost_fn=1, cost_fp=1, pi_T=0.5,  K=0.0, c=1.0)
    
    plt.savefig('Graph/SVM/Quadratic/5FoldGAU.png' )

#--------------------------

def print_graphs_Polinomial_SVM_Cs_k_c(DTR, LTR, k ):

    def oneGraphSingleFold(data, prior, cost_fn, cost_fp, pi_T, K, c):
        print("working on single fold k = ", K, "c = ", c)
        exps = numpy.linspace(-2,2, 5)
        Cs = 10** exps
        minDCFs = 0 * exps
        for i in range (Cs.size):
            C= Cs[i]
            minDCFs[i],_,_ = model_validation.singleFold_DCF(data, LTR, SVMClassifier.Polinomial_SVM_computeLogLikelihoods, prior , cost_fn, cost_fp, [pi_T, C, c, K])
        
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
            minDCFs[i],_,_ = model_validation.k_cross_DCF(data, LTR,k, SVMClassifier.Polinomial_SVM_computeLogLikelihoods, prior , cost_fn, cost_fp, [pi_T, C, c, K])
        
        lb = "minDCF (k="+ str(K) +" c= "+ str(c)+ ")"
        plt.plot(Cs, minDCFs, label=lb)
        plt.legend()
        print("DONE")

    normalizedFeatures = preprocessing.Z_normalization(DTR)
    gaussianizedFeatures = preprocessing.gaussianization(DTR)

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

#--------------------------

def print_table_Quadratic_SVM_minDCF(DTR, LTR, prior, cost_fn, cost_fp, k): 

    def quadratic_SVM_minDCF(data, C, c, K):
        
        pi_T = 0.5
        minDCF,_,_ = model_validation.singleFold_DCF(data, LTR, SVMClassifier.Polinomial_SVM_computeLogLikelihoods, prior , cost_fn, cost_fp, [pi_T, C, c, K])
        print("[5-Folds]  -  C= ", C, ", pi_T=0.5, c= ", c, " k = ",K ,"  : ",minDCF)  

        
        pi_T = 0.1
        minDCF,_,_ = model_validation.k_cross_DCF(data, LTR,k, SVMClassifier.SVM_computeLogLikelihoods, prior , cost_fn, cost_fp, [pi_T, C, c, K])
        print("[5-Folds]  -  C= ", C, ", pi_T=0.1, c= ", c, " k = ",K ,"  : ",minDCF)

        
        pi_T = 0.9
        minDCF,_,_ = model_validation.k_cross_DCF(data, LTR,k, SVMClassifier.SVM_computeLogLikelihoods, prior , cost_fn, cost_fp, [pi_T, C, c, K])
        print("[5-Folds]  -  C= ", C, ", pi_T=0.9, c= ", c, " k = ",K, "  : ",minDCF)


        
        N = LTR.size #tot number of samples
        n_T = (1*(LTR==1)).sum() #num of samples belonging to the true class
        n_F = (1*(LTR==0)).sum() #num of samples belonging to the false class
        pi_emp_T = n_T / N

        pi_T = pi_emp_T
        minDCF,_,_ = model_validation.k_cross_DCF(data, LTR,k, SVMClassifier.SVM_computeLogLikelihoods, prior , cost_fn, cost_fp,[pi_T, C, c, K])
        print("[5-Folds]  - C= ", C, ", pi_T=pi_emp_T, c= ", c, " k = ",K , "  : ",minDCF)


        print()

    def fun_parametri(C,c, K):
        gaussianizedFeatures = preprocessing.gaussianization(DTR)
        normalizedFeatures = preprocessing.Z_normalization(DTR)
        
        
        print("PARAMETRI: (C = " + str(C) + " c= "+ str(c) + "K= " + str(K)+ ")") 

        #------------------------RAW FEATURES -----------------
        print("*** minDCF - RAW FEATURES ***")
        quadratic_SVM_minDCF(normalizedFeatures, C=C, c=c, K=K)

        #--------------- GAUSSIANIZED FEATURES-------------------------
        print("*** minDCF - GAUSSIANIZED FEATURES  ***")
        quadratic_SVM_minDCF(gaussianizedFeatures ,C=C, c=c, K=K)


        print("************************************************")
    
    fun_parametri(0.1,1,0)
    #fun_parametri(100,1,0) 
    #fun_parametri(0.1,1,1)

#--------------------------

def print_graphs_RBF_SVM_Cs(DTR, LTR, k):

    def oneGraphKFold(data, prior, cost_fn, cost_fp, pi_T, loglam):
        print("working on k fold loglam = ", loglam)
        exps = numpy.linspace(-1,2, 10)
        Cs = 10** exps
        minDCFs = 0 * exps
        for i in range (Cs.size):
            C= Cs[i]
            minDCFs[i],_,_ = model_validation.k_cross_DCF(data, LTR,k, SVMClassifier.RBF_SVM_computeLogLikelihoods, prior , cost_fn, cost_fp, [pi_T, C, 10**loglam])
        
        lb = " (log(lam)="+ str(loglam) +")"
        plt.plot(Cs, minDCFs, label=lb)
        plt.legend()
        print("DONE")

    normalizedFeatures = preprocessing.Z_normalization(DTR)
    gaussianizedFeatures = preprocessing.gaussianization(DTR)

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

#--------------------------

def print_table_RBF_SVM_minDCF(DTR, LTR, prior, cost_fn, cost_fp, k): #TODO

    def RBF_SVM_minDCF(data, C, loglam):
        
        pi_T = 0.5
        minDCF,_,_ = model_validation.k_cross_DCF(data, LTR,k, SVMClassifier.RBF_SVM_computeLogLikelihoods, prior , cost_fn, cost_fp, [pi_T, C, 10**loglam])
        print("[5-Folds]  -  C= ", C, ", loglam= ", loglam, " pi_T=0.5: ",minDCF)   

        
        pi_T = 0.1
        minDCF,_,_ = model_validation.k_cross_DCF(data, LTR,k, SVMClassifier.RBF_SVM_computeLogLikelihoods, prior , cost_fn, cost_fp, [pi_T, C, 10**loglam])
        print("[5-Folds]  -  C= ", C, ", loglam= ", loglam, " pi_T=0.1: ",minDCF)   

        
        pi_T = 0.9
        minDCF,_,_ = model_validation.k_cross_DCF(data, LTR,k, SVMClassifier.RBF_SVM_computeLogLikelihoods, prior , cost_fn, cost_fp, [pi_T, C, 10**loglam])
        print("[5-Folds]  -  C= ", C, ", loglam= ", loglam, " pi_T=0.9: ",minDCF)   


        
        N = LTR.size #tot number of samples
        n_T = (1*(LTR==1)).sum() #num of samples belonging to the true class
        n_F = (1*(LTR==0)).sum() #num of samples belonging to the false class
        pi_emp_T = n_T / N

        pi_T = pi_emp_T
        minDCF,_,_ = model_validation.k_cross_DCF(data, LTR,k, SVMClassifier.RBF_SVM_computeLogLikelihoods, prior , cost_fn, cost_fp, [pi_T, C, 10**loglam])
        print("[5-Folds]  -  C= ", C, ", loglam= ", loglam, " pi_T=pi_emp_T: ",minDCF)   


        print()

    def fun_parametri(C,loglam):
        gaussianizedFeatures = preprocessing.gaussianization(DTR)
        normalizedFeatures = preprocessing.Z_normalization(DTR)
        
        
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

#--------------------------

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
        plt.legend(loc="lower left")

        plt.savefig('Graph/GMM/'+title+'.png' )


    def GMM_compute_DCFs(DTR, LTR, k, covariance_type, prior, cost_fn, cost_fp):
        gmm_comp = [1,2,4,8,16,32]

        raw_minDCFs = []
        gau_minDCFs = []

        normalized_features = preprocessing.Z_normalization(DTR)
        gaussianizedFeatures = preprocessing.gaussianization(DTR)

        constrained=True
        psi=0.01
        alpha=0.1
        delta_l=10**(-6)
    
        print("************************" + covariance_type + "*************************")
        for i in range(len(gmm_comp)):
            params = [constrained, psi, covariance_type, alpha, gmm_comp[i],delta_l]
            print("-------> working on raw data, comp= ", gmm_comp[i])
            # Raw features
            raw_minDCFs_i,_,_ = model_validation.k_cross_DCF(normalized_features, LTR, k, gaussian_mixture_models.GMM_computeLogLikelihoodRatios, prior , cost_fn, cost_fp, params)
            print("RAW DATA, num components = " + str(gmm_comp[i]) + ", minDCF = " + str(raw_minDCFs_i) )
            # Gaussianized features
            print("-------> working on gauss data, comp= ", gmm_comp[i])
            gau_minDCFs_i,_,_ = model_validation.k_cross_DCF(gaussianizedFeatures, LTR,k, gaussian_mixture_models.GMM_computeLogLikelihoodRatios, prior , cost_fn, cost_fp, params)
            print("GAUSS DATA, num components = " + str(gmm_comp[i]) + ", minDCF = " + str(gau_minDCFs_i) )
            raw_minDCFs.append(raw_minDCFs_i)
            gau_minDCFs.append(gau_minDCFs_i)
            print()    
        
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

#--------------------------

def print_table_comparison_DCFs(DTR, LTR, k):

    def actDCF_minDCF(data, llr_calculator, params):
            prior=0.5
            cost_fn=1
            cost_fp=1
            min_DCF_LR, act_DCF_LR,_ = model_validation.k_cross_DCF(data, LTR, k, llr_calculator, prior , cost_fn, cost_fp, params)
            print("[5-Folds]  -  prior= 0.5  minDCF: ",min_DCF_LR, " actDCF= ",act_DCF_LR)

            prior=0.1
            cost_fn=1
            cost_fp=1
            min_DCF_LR, act_DCF_LR,_ = model_validation.k_cross_DCF(data, LTR, k, llr_calculator, prior , cost_fn, cost_fp, params)
            print("[5-Folds]  -  prior= 0.1  minDCF: ",min_DCF_LR, " actDCF= ",act_DCF_LR) 

            prior=0.9
            cost_fn=1
            cost_fp=1
            min_DCF_LR, act_DCF_LR,_ = model_validation.k_cross_DCF(data, LTR, k, llr_calculator, prior , cost_fn, cost_fp, params)
            print("[5-Folds]  -  prior= 0.9  minDCF: ",min_DCF_LR, " actDCF= ",act_DCF_LR) 

            print()


    #------------------------FIRST MODEL ----------------- 

    print(" RBF SVM, C=1, lam=1, pi_T =0.5 raw features ")
    lam = 1
    pi_T = 0.5
    C= 1
    actDCF_minDCF(preprocessing.Z_normalization(DTR), SVMClassifier.RBF_SVM_computeLogLikelihoods,[pi_T, C, lam] )

    #--------------- SECOND MODEL-------------------------

    print(" Quad SVM, C=0.1, pi_T =0.5, c=1,K=0 raw features  ")
    C=0.1
    pi_T=0.5
    c=1
    K=0
    actDCF_minDCF(preprocessing.Z_normalization(DTR), SVMClassifier.Polinomial_SVM_computeLogLikelihoods,[pi_T,C,c,K])

#--------------------------
def print_err_bayes_plots(data, L, k, llr_calculators, other_params, titles, colors):
    plt.figure()
    plt.title("Bayes Error Plot")
    plt.xlabel("prior log odds")
    plt.ylabel("DCF")
    for i in range (len(llr_calculators)):
        print("Working on calculator "+ str(i))
        model_validation.bayes_error_plot(data[i], L, k, llr_calculators[i], other_params[i], titles[i], colors[i] )
        print("DONE")
    plt.savefig('Graph/Error_Bayes_Plots/EBP1.png' )

#--------------------------

def print_treshold_estimated_table(data, LTR, prior, cost_fn, cost_fp, k, llr_calculator, otherParams, title):
    
    minDCF, actDCF_th, actDCF_opt, optimal_treshold = model_validation.compute_DCF_with_optimal_treshold(data, LTR, k, llr_calculator, otherParams, prior, cost_fn, cost_fp )

    print(title + ":")
    print("minDCF = ", minDCF)
    print("actual theoretical DCF = ", actDCF_th)
    print("actual optimal DCF = ", actDCF_opt)
    
    return optimal_treshold

#--------------------------
def print_all(DTR, LTR, k):
    
    '''
    ### -- MVG CLASSIFIERS
    print("********************* MVG TABLE ************************************")
    print("------> pi = 0.5")
    print_table_MVG_classifiers_minDCF(DTR, LTR, prior=0.3, cost_fn=1, cost_fp=1, k=k)
    print()
    
    print("------> pi = 0.9")
    print_table_MVG_classifiers_minDCF(DTR, LTR, prior=0.9, cost_fn=1, cost_fp=1, k=k)
    print()

    print("------> pi = 0.1")
    print_table_MVG_classifiers_minDCF(DTR, LTR, prior=0.1, cost_fn=1, cost_fp=1, k=k)
    print()
    print("********************************************************************")
    
    ### -- LINEAR LOGISTIC REGRESSION
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

    
    ### -- QUADRATIC LOGISTIC REGRESSION

    print("********************* quadratic LR GRAPHS ************************************")
    print_graphs_quadratic_LR_lambdas(DTR, LTR,  k)
    print("********************************************************************")
    
   
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
    
    
    
    ### -- LINEAR SVM
    
    print("********************* SVM GRAPHS ************************************")
    print_graphs_SVM_Cs(DTR, LTR, k=k )
    print("********************************************************************")
    
    print("********************* SVM TABLES ************************************")
    print("------> applicazione con prior = 0.5")
    print_table_SVM_minDCF(DTR, LTR, prior=0.5, cost_fn=1, cost_fp=1, k=k )
    print("------> applicazione con prior = 0.9")
    print_table_SVM_minDCF(DTR, LTR, prior=0.9, cost_fn=1, cost_fp=1, k=k )
    print("------> applicazione con prior = 0.1")
    print_table_SVM_minDCF(DTR, LTR, prior=0.1, cost_fn=1, cost_fp=1, k=k )
    print("********************************************************************")

    
    
    ### -- QUADRATIC SVM

    print("********************* quadratic SVM GRAPHS changing C,k,c ************************************")
    print_graphs_Polinomial_SVM_Cs_k_c(DTR, LTR, k=k )
    print("********************************************************************")
    '''
    print("********************* quadratic SVM GRAPHS ************************************")
    print_graphs_Polinomial_SVM_Cs(DTR, LTR, k=k )
    print("********************************************************************")
    '''
    print("********************* quadratic SVM TABLES ************************************")
    print("------> applicazione con prior = 0.5")
    print_table_Quadratic_SVM_minDCF(DTR, LTR, prior=0.5, cost_fn=1, cost_fp=1, k=k )
    print("------> applicazione con prior = 0.9")
    print_table_Quadratic_SVM_minDCF(DTR, LTR, prior=0.9, cost_fn=1, cost_fp=1, k=k )
    print("------> applicazione con prior = 0.1")
    print_table_Quadratic_SVM_minDCF(DTR, LTR, prior=0.1, cost_fn=1, cost_fp=1, k=k )
    print("********************************************************************")
    
    
    ### RBF

    print("********************* RBF SVM GRAPHS ************************************")
    print_graphs_RBF_SVM_Cs(DTR, LTR, k=k )
    print("********************************************************************")
    

    print("********************* RBF SVM TABLES ************************************")
    print("------> applicazione con prior = 0.5")
    print_table_RBF_SVM_minDCF(DTR, LTR, prior=0.5, cost_fn=1, cost_fp=1, k=k )
    print("------> applicazione con prior = 0.9")
    print_table_RBF_SVM_minDCF(DTR, LTR, prior=0.9, cost_fn=1, cost_fp=1, k=k )
    print("------> applicazione con prior = 0.1")
    print_table_RBF_SVM_minDCF(DTR, LTR, prior=0.1, cost_fn=1, cost_fp=1, k=k )
    print("********************************************************************")
    

    ### GMM
    print_graphs_GMM_minDCF(DTR, LTR, k)


    '''
    ## COMPARISON BETWEEN ACT DCF AND MIN DCF OF THE CHOSEN MODELS

    print_table_comparison_DCFs(DTR, LTR, k=k)

    #error bayes plot

    pi_T1 = 0.5
    C1= 1
    lam1 = 1

    pi_T2=0.5
    C2=0.1
    c2=1
    K2=0


    data = [ preprocessing.Z_normalization(DTR), preprocessing.Z_normalization(DTR)]
    llr_calculators = [SVMClassifier.RBF_SVM_computeLogLikelihoods,SVMClassifier.Polinomial_SVM_computeLogLikelihoods ]
    other_params = [[pi_T1,C1,lam1], [pi_T2,C2,c2,K2]]
    titles = ["RBF SVM RAW", "Quad SVM RAW"]
    colors = ["r", "b"]
    
    print("************ PRINT BAYES ERROR PLOT******************")
    print_err_bayes_plots(data, LTR, k, llr_calculators, other_params, titles, colors)
    print("*****************************************************")


    print("------>Treshold estimated table:")
    print()
    print("------> applicazione prior = 0.5")
    prior = 0.5
    print_treshold_estimated_table(data[0], LTR, prior, 1, 1, k, llr_calculators[0], other_params[0], titles[0])
    print_treshold_estimated_table(data[1], LTR, prior, 1, 1, k, llr_calculators[1], other_params[1], titles[1])
    print()

    print("------> applicazione prior = 0.1")
    prior = 0.1
    print_treshold_estimated_table(data[0], LTR, prior, 1, 1, k, llr_calculators[0], other_params[0], titles[0])
    print_treshold_estimated_table(data[1], LTR, prior, 1, 1, k, llr_calculators[1], other_params[1], titles[1])
    print()


    print("------> applicazione prior = 0.9")
    prior = 0.9
    print_treshold_estimated_table(data[0], LTR, prior, 1, 1, k, llr_calculators[0], other_params[0], titles[0])
    print_treshold_estimated_table(data[1], LTR, prior, 1, 1, k, llr_calculators[1], other_params[1], titles[1])
    print()

    

#--------------------------
