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


def print_table_MVG_classifiers_minDCF(DTR, LTR, prior, cost_fn, cost_fp, k, eval_data):

    def MVG_Classifiers_minDCF(data, eval_data):
        #Full_Cov 
        min_DCF_MVG,_,_ = model_validation.singleFold_DCF(data, LTR, MVGclassifiers.MVG_logLikelihoodRatios, prior , cost_fn, cost_fp, eval_data=eval_data)
        print("[Single Fold] -  MVG: ",min_DCF_MVG)  
        min_DCF_MVG,_,_ = model_validation.k_cross_DCF(data, LTR, k, MVGclassifiers.MVG_logLikelihoodRatios, prior , cost_fn, cost_fp, eval_data=eval_data)
        print("[5-Folds]  -  MVG: ",min_DCF_MVG)  

        #Diag_Cov == Naive
        min_DCF_Diag_Cov,_,_ = model_validation.singleFold_DCF(data, LTR, MVGclassifiers.NAIVE_logLikelihoodRatios, prior , cost_fn, cost_fp, eval_data=eval_data)
        print("[Single Fold]  - MVG with Diag cov: ",min_DCF_Diag_Cov)
        min_DCF_Diag_Cov,_,_ = model_validation.k_cross_DCF(data, LTR,k, MVGclassifiers.NAIVE_logLikelihoodRatios, prior , cost_fn, cost_fp, eval_data=eval_data)
        print("[5- Fold] - MVG with Diag cov: ",min_DCF_Diag_Cov)

        #Tied
        min_DCF_Tied,_,_ = model_validation.singleFold_DCF(data, LTR, MVGclassifiers.TIED_logLikelihoodRatios, prior , cost_fn, cost_fp, eval_data=eval_data)
        print("[Single Fold] - Tied MVG: ",min_DCF_Tied)
        min_DCF_Tied,_,_ = model_validation.k_cross_DCF(data, LTR,k, MVGclassifiers.TIED_logLikelihoodRatios, prior , cost_fn, cost_fp, eval_data=eval_data)
        print("[5- Fold] - Tied MVG: ",min_DCF_Tied)

        #Tied Diag_Cov
        min_DCF_Tied_Diag_Cov,_,_ = model_validation.singleFold_DCF(data, LTR, MVGclassifiers.TIED_DIAG_COV_logLikelihoodRatios, prior , cost_fn, cost_fp, eval_data=eval_data)
        print("[Single Fold] - Tied MVG with Diag Cov: ",min_DCF_Tied_Diag_Cov)
        min_DCF_Tied_Diag_Cov,_,_ = model_validation.k_cross_DCF(data, LTR, k,  MVGclassifiers.TIED_DIAG_COV_logLikelihoodRatios, prior , cost_fn, cost_fp, eval_data=eval_data)
        print("[5 Fold] - Tied MVG with Diag Cov: ",min_DCF_Tied_Diag_Cov)

        print()

    DTE = eval_data[0]
    LTE = eval_data[1]

    #!!! normalization is important before PCA
    normalized_data = preprocessing.Z_normalization(DTR)
    normalized_data_eval = preprocessing.Z_normalization(DTE)
    gaussianizedFeatures = preprocessing.gaussianization(DTR)
    gaussianizedFeatures_eval = preprocessing.gaussianizationEval(DTR, DTE)
    principal_components10, principal_components10_eval = redTec.PCA_evaluation(normalized_data, 10, normalized_data_eval)
    principal_components9, principal_components9_eval = redTec.PCA_evaluation(normalized_data, 9, normalized_data_eval)
    
    #------------------------RAW FEATURES (normalized) -----------------
    print("*** minDCF - RAW (normalized) FEATURES - NO PCA ***")
    MVG_Classifiers_minDCF(normalized_data, [normalized_data_eval, LTE])
    

    #------------------------RAW FEATURES (normalized) WITH PCA = 10 --------------------
    print("*** minDCF - RAW (normalized) FEATURES -  PCA (m=10) ***")
    MVG_Classifiers_minDCF(principal_components10, [principal_components10_eval, LTE])       


    #------------------------RAW FEATURES (normalized) WITH PCA = 9 --------------------
    
    print("*** minDCF - RAW (normalized) FEATURES -  PCA (m=9) ***")
    MVG_Classifiers_minDCF(principal_components9, [principal_components9_eval, LTE])    


    ## Z --> PCA --> GAUSS
    #--------------- GAUSSIANIZED FEATURES-------------------------
    print("*** minDCF - GAUSSIANIZED FEATURES - NO PCA ***")
    MVG_Classifiers_minDCF(gaussianizedFeatures, [gaussianizedFeatures_eval, LTE])


    #------------------------GAUSSIANIZED FEATURES WITH PCA = 10 --------------------
    print("*** minDCF - GAUSSIANIZED FEATURES -  PCA m=10 ***")
    gaussianized_principal_components_10 = preprocessing.gaussianization(principal_components10)
    gaussianized_principal_components_10_eval = preprocessing.gaussianizationEval(principal_components10, principal_components10_eval)

    MVG_Classifiers_minDCF(gaussianized_principal_components_10, [gaussianized_principal_components_10_eval, LTE])     


    #------------------------GAUSSIANIZED FEATURES WITH PCA = 9 --------------------
    print("*** minDCF - GAUSSIANIZED FEATURES -  PCA m=9 ***")
    gaussianized_principal_components_9 = preprocessing.gaussianization(principal_components9)
    gaussianized_principal_components_9_eval = preprocessing.gaussianizationEval(principal_components9, principal_components9_eval)
    MVG_Classifiers_minDCF(gaussianized_principal_components_9, [gaussianized_principal_components_9_eval, LTE])

#--------------------------

def print_table_LR_minDCF(DTR, LTR, prior, cost_fn, cost_fp, k, eval_data):

    def LR_minDCF(data, eval_data):
            
        lam = 10**(-3)
        pi_T = 0.5
        min_DCF_LR,_,_ = model_validation.k_cross_DCF(data, LTR, k, logisticRegression.LR_logLikelihoodRatios, prior , cost_fn, cost_fp, [lam, pi_T], eval_data=eval_data)
        print("[5-Folds]  -  lam = 10^-3, pi_T = 0.5: ",min_DCF_LR)  

        pi_T = 0.1
        min_DCF_LR,_,_ = model_validation.k_cross_DCF(data, LTR, k, logisticRegression.LR_logLikelihoodRatios, prior , cost_fn, cost_fp, [lam, pi_T], eval_data=eval_data)
        print("[5-Folds]  -  lam = 10^-3, pi_T = 0.1: ",min_DCF_LR)

        pi_T = 0.9
        min_DCF_LR,_,_ = model_validation.k_cross_DCF(data, LTR, k, logisticRegression.LR_logLikelihoodRatios, prior , cost_fn, cost_fp, [lam, pi_T], eval_data=eval_data)
        print("[5-Folds]  -  lam = 10^-3, pi_T = 0.9: ",min_DCF_LR)
        

        N = LTR.size #tot number of samples
        n_T = (1*(LTR==1)).sum() #num of samples belonging to the true class
        pi_emp_T = n_T / N

        pi_T = pi_emp_T
        
        min_DCF_LR,_,_ = model_validation.k_cross_DCF(data, LTR, k, logisticRegression.LR_logLikelihoodRatios, prior , cost_fn, cost_fp, [lam, pi_T])
        print("[5-Folds]  -  lam = 10^-3, pi_T = pi_emp_T: ",min_DCF_LR)

        print()

    DTE = eval_data[0]
    LTE = eval_data[1]
    
    #------------------------RAW FEATURES -----------------
    print("*** minDCF - RAW FEATURES ***")
    LR_minDCF(preprocessing.Z_normalization(DTR), [preprocessing.Z_normalization(DTE), LTE])

    #--------------- GAUSSIANIZED FEATURES-------------------------
    gaussianizedFeatures = preprocessing.gaussianization(DTR)

    print("*** minDCF - GAUSSIANIZED FEATURES  ***")
    LR_minDCF(gaussianizedFeatures, [preprocessing.gaussianizationEval(DTR, DTE), LTE])

#--------------------------

def print_table_Quadratic_LR_minDCF(DTR, LTR, prior, cost_fn, cost_fp, k, eval_data):

    def Quad_LR_minDCF(data, eval_data):
        lam = 10**(-3)
        
        pi_T = 0.5
        min_DCF_LR,_,_ = model_validation.k_cross_DCF(data, LTR, k, logisticRegression.Quadratic_LR_logLikelihoodRatios, prior , cost_fn, cost_fp, [lam, pi_T], eval_data=eval_data)
        print("[5-Folds]  -  lam = 10^-3, pi_T = 0.5: ",min_DCF_LR)  

        pi_T = 0.1
        min_DCF_LR,_,_ = model_validation.k_cross_DCF(data, LTR, k, logisticRegression.Quadratic_LR_logLikelihoodRatios, prior , cost_fn, cost_fp, [lam, pi_T], eval_data=eval_data)
        print("[5-Folds]  -  lam = 10^-3, pi_T = 0.1: ",min_DCF_LR)

        pi_T = 0.9
        min_DCF_LR,_,_ = model_validation.k_cross_DCF(data, LTR, k, logisticRegression.Quadratic_LR_logLikelihoodRatios, prior , cost_fn, cost_fp, [lam, pi_T], eval_data=eval_data)
        print("[5-Folds]  -  lam = 10^-3, pi_T = 0.9: ",min_DCF_LR)
        

        N = LTR.size #tot number of samples
        n_T = (1*(LTR==1)).sum() #num of samples belonging to the true class
        pi_emp_T = n_T / N

        pi_T = pi_emp_T
        min_DCF_LR,_,_ = model_validation.k_cross_DCF(data, LTR, k, logisticRegression.Quadratic_LR_logLikelihoodRatios, prior , cost_fn, cost_fp, [lam, pi_T], eval_data=eval_data)
        print("[5-Folds]  -  lam = 10^-3, pi_T = pi_emp_T: ",min_DCF_LR)
        
        print()

    DTE = eval_data[0]
    LTE = eval_data[1]

    #------------------------RAW FEATURES -----------------
    print("*** minDCF - RAW FEATURES ***")
    Quad_LR_minDCF(preprocessing.Z_normalization(DTR), [preprocessing.Z_normalization(DTE), LTE])

    #--------------- GAUSSIANIZED FEATURES-------------------------
    gaussianizedFeatures = preprocessing.gaussianization(DTR)

    print("*** minDCF - GAUSSIANIZED FEATURES  ***")
    Quad_LR_minDCF(gaussianizedFeatures, [preprocessing.gaussianizationEval(DTR,DTE), LTE])

#--------------------------

def print_table_SVM_minDCF(DTR, LTR, prior, cost_fn, cost_fp, k, eval_data):

    def linear_SVM_minDCF(data, eval_data):
        C = 0.1
        pi_T = 0.5
        minDCF,_,_ = model_validation.k_cross_DCF(data, LTR,k, SVMClassifier.SVM_computeLogLikelihoods, prior , cost_fn, cost_fp, [pi_T, C], eval_data=eval_data)
        print("[5-Folds]  -  C= 0.1, pi_T=0.5: ",minDCF)  

        C = 0.1
        pi_T = 0.1
        minDCF,_,_ = model_validation.k_cross_DCF(data, LTR,k, SVMClassifier.SVM_computeLogLikelihoods, prior , cost_fn, cost_fp, [pi_T, C], eval_data=eval_data)
        print("[5-Folds]  -  C= 0.1, pi_T=0.1: ",minDCF)

        C = 0.1
        pi_T = 0.9
        minDCF,_,_ = model_validation.k_cross_DCF(data, LTR,k, SVMClassifier.SVM_computeLogLikelihoods, prior , cost_fn, cost_fp, [pi_T, C], eval_data=eval_data)
        print("[5-Folds]  -  C= 0.1, pi_T=0.9: ",minDCF)

        #unbalanced application
        C = 0.1
        pi_T = -1
        minDCF,_,_ = model_validation.k_cross_DCF(data, LTR,k, SVMClassifier.SVM_computeLogLikelihoods, prior , cost_fn, cost_fp, [pi_T, C], eval_data=eval_data)
        print("[5-Folds]  -  C= 0.1, pi_T=pi_emp_T: ",minDCF)

        print()

    DTE = eval_data[0]
    LTE = eval_data[1]

    #------------------------RAW FEATURES -----------------
    print("*** minDCF - RAW FEATURES ***")
    linear_SVM_minDCF(preprocessing.Z_normalization(DTR), [preprocessing.Z_normalization(DTE), LTE])

    #--------------- GAUSSIANIZED FEATURES-------------------------
    gaussianizedFeatures = preprocessing.gaussianization(DTR)

    print("*** minDCF - GAUSSIANIZED FEATURES  ***")
    linear_SVM_minDCF(gaussianizedFeatures, [preprocessing.gaussianizationEval(DTR, DTE), LTE])

#--------------------------

def print_table_Quadratic_SVM_minDCF(DTR, LTR, prior, cost_fn, cost_fp, k, eval_data): 

    def quadratic_SVM_minDCF(data, C, c, K, eval_data):
        
        pi_T = 0.5
        minDCF,_,_ = model_validation.singleFold_DCF(data, LTR, SVMClassifier.Polinomial_SVM_computeLogLikelihoods, prior , cost_fn, cost_fp, [pi_T, C, c, K], eval_data=eval_data)
        print("[5-Folds]  -  C= ", C, ", pi_T=0.5, c= ", c, " k = ",K ,"  : ",minDCF)  

        
        pi_T = 0.1
        minDCF,_,_ = model_validation.k_cross_DCF(data, LTR,k, SVMClassifier.SVM_computeLogLikelihoods, prior , cost_fn, cost_fp, [pi_T, C, c, K], eval_data=eval_data)
        print("[5-Folds]  -  C= ", C, ", pi_T=0.1, c= ", c, " k = ",K ,"  : ",minDCF)

        
        pi_T = 0.9
        minDCF,_,_ = model_validation.k_cross_DCF(data, LTR,k, SVMClassifier.SVM_computeLogLikelihoods, prior , cost_fn, cost_fp, [pi_T, C, c, K], eval_data=eval_data)
        print("[5-Folds]  -  C= ", C, ", pi_T=0.9, c= ", c, " k = ",K, "  : ",minDCF)


        
        N = LTR.size #tot number of samples
        n_T = (1*(LTR==1)).sum() #num of samples belonging to the true class
        n_F = (1*(LTR==0)).sum() #num of samples belonging to the false class
        pi_emp_T = n_T / N

        pi_T = pi_emp_T
        minDCF,_,_ = model_validation.k_cross_DCF(data, LTR,k, SVMClassifier.SVM_computeLogLikelihoods, prior , cost_fn, cost_fp,[pi_T, C, c, K], eval_data=eval_data)
        print("[5-Folds]  - C= ", C, ", pi_T=pi_emp_T, c= ", c, " k = ",K , "  : ",minDCF)


        print()

    def fun_parametri(C,c, K, eval_data):
        DTE = eval_data[0]
        LTE = eval_data[1]

        gaussianizedFeatures = preprocessing.gaussianization(DTR)
        normalizedFeatures = preprocessing.Z_normalization(DTR)
        
        
        print("PARAMETRI: (C = " + str(C) + " c= "+ str(c) + "K= " + str(K)+ ")") 

        #------------------------RAW FEATURES -----------------
        print("*** minDCF - RAW FEATURES ***")
        quadratic_SVM_minDCF(normalizedFeatures, C=C, c=c, K=K, eval_data=[preprocessing.Z_normalization(DTE), LTE])

        #--------------- GAUSSIANIZED FEATURES-------------------------
        print("*** minDCF - GAUSSIANIZED FEATURES  ***")
        quadratic_SVM_minDCF(gaussianizedFeatures ,C=C, c=c, K=K, eval_data=[preprocessing.gaussianizationEval(DTR, DTE), LTE])


        print("************************************************")
    
    fun_parametri(10,1,0, eval_data)
    fun_parametri(100,1,0, eval_data)
    fun_parametri(0.1,1,1, eval_data)

#--------------------------

def print_table_RBF_SVM_minDCF(DTR, LTR, prior, cost_fn, cost_fp, k, eval_data): #TODO

    def RBF_SVM_minDCF(data, C, loglam, eval_data):
        
        pi_T = 0.5
        minDCF,_,_ = model_validation.k_cross_DCF(data, LTR,k, SVMClassifier.RBF_SVM_computeLogLikelihoods, prior , cost_fn, cost_fp, [pi_T, C, 10**loglam], eval_data=eval_data)
        print("[5-Folds]  -  C= ", C, ", loglam= ", loglam, " pi_T=0.5: ",minDCF)   

        
        pi_T = 0.1
        minDCF,_,_ = model_validation.k_cross_DCF(data, LTR,k, SVMClassifier.RBF_SVM_computeLogLikelihoods, prior , cost_fn, cost_fp, [pi_T, C, 10**loglam], eval_data=eval_data)
        print("[5-Folds]  -  C= ", C, ", loglam= ", loglam, " pi_T=0.1: ",minDCF)   

        
        pi_T = 0.9
        minDCF,_,_ = model_validation.k_cross_DCF(data, LTR,k, SVMClassifier.RBF_SVM_computeLogLikelihoods, prior , cost_fn, cost_fp, [pi_T, C, 10**loglam], eval_data=eval_data)
        print("[5-Folds]  -  C= ", C, ", loglam= ", loglam, " pi_T=0.9: ",minDCF)   


        
        N = LTR.size #tot number of samples
        n_T = (1*(LTR==1)).sum() #num of samples belonging to the true class
        n_F = (1*(LTR==0)).sum() #num of samples belonging to the false class
        pi_emp_T = n_T / N

        pi_T = pi_emp_T
        minDCF,_,_ = model_validation.k_cross_DCF(data, LTR,k, SVMClassifier.RBF_SVM_computeLogLikelihoods, prior , cost_fn, cost_fp, [pi_T, C, 10**loglam], eval_data=eval_data)
        print("[5-Folds]  -  C= ", C, ", loglam= ", loglam, " pi_T=pi_emp_T: ",minDCF)   


        print()

    def fun_parametri(C,loglam, eval_data):
        DTE = eval_data[0]
        LTE = eval_data[1]

        gaussianizedFeatures = preprocessing.gaussianization(DTR)
        normalizedFeatures = preprocessing.Z_normalization(DTR)
        
        
        print("PARAMETRI: (C = " + str(C) + " loglam= "+ str(loglam)+ ")") 

        #------------------------RAW FEATURES -----------------
        print("*** minDCF - RAW FEATURES ***")
        RBF_SVM_minDCF(normalizedFeatures, C=C, loglam=loglam, eval_data=[preprocessing.Z_normalization(DTE), LTE])

        #--------------- GAUSSIANIZED FEATURES-------------------------
        print("*** minDCF - GAUSSIANIZED FEATURES  ***")
        RBF_SVM_minDCF(gaussianizedFeatures ,C=C, loglam=loglam, eval_data=[preprocessing.gaussianizationEval(DTR, DTE), LTE])


        print("************************************************")
    
    fun_parametri(1, 0, eval_data)
    fun_parametri(0.5, 0, eval_data)

#--------------------------

def print_table_GMM_minDCF(DTR, LTR, k, eval_data):

    def GMM_compute_DCFs(DTR, LTR, k, covariance_type, prior, cost_fn, cost_fp, eval_data):

        DTE = eval_data[0]
        LTE = eval_data[1]

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
            raw_minDCFs_i,_,_ = model_validation.k_cross_DCF(normalized_features, LTR, k, gaussian_mixture_models.GMM_computeLogLikelihoodRatios, prior , cost_fn, cost_fp, params, [preprocessing.Z_normalization(DTE), LTE])
            print("RAW DATA, num components = " + str(gmm_comp[i]) + ", minDCF = " + str(raw_minDCFs_i) )
            # Gaussianized features
            print("-------> working on gauss data, comp= ", gmm_comp[i])
            gau_minDCFs_i,_,_ = model_validation.k_cross_DCF(gaussianizedFeatures, LTR,k, gaussian_mixture_models.GMM_computeLogLikelihoodRatios, prior , cost_fn, cost_fp, params, [preprocessing.gaussianizationEval(DTR, DTE), LTE])
            print("GAUSS DATA, num components = " + str(gmm_comp[i]) + ", minDCF = " + str(gau_minDCFs_i) )
            raw_minDCFs.append(raw_minDCFs_i)
            gau_minDCFs.append(gau_minDCFs_i)
            print()    
        
        raw_minDCFs=numpy.array(raw_minDCFs)
        gau_minDCFs=numpy.array(gau_minDCFs)
        return raw_minDCFs, gau_minDCFs, gmm_comp


    #### Full Cov
    covariance_type = "Full"
    raw_minDCFs, gau_minDCFs, gmm_comp = GMM_compute_DCFs(DTR, LTR, k, covariance_type, 0.5, 1, 1, eval_data)

    #### Diagonal Cov
    covariance_type = "Diagonal"
    raw_minDCFs, gau_minDCFs, gmm_comp = GMM_compute_DCFs(DTR, LTR, k, covariance_type, 0.5, 1, 1, eval_data)

    #### Diagonal Cov
    covariance_type = "Tied"
    raw_minDCFs, gau_minDCFs, gmm_comp = GMM_compute_DCFs(DTR, LTR, k, covariance_type, 0.5, 1, 1, eval_data)
    
    #### Diagonal Cov
    covariance_type = "Tied Diagonal"
    raw_minDCFs, gau_minDCFs, gmm_comp = GMM_compute_DCFs(DTR, LTR, k, covariance_type, 0.5, 1, 1, eval_data)



#--------------------------
def print_all(DTR, LTR, DEV, LEV, k):
    
    eval_data = [DEV, LEV]

    print("************************EVALUATION RESULT***********************")
    ### -- MVG CLASSIFIERS
    '''
    print("********************* MVG TABLE ************************************")
    print("------> pi = 0.5")
    print_table_MVG_classifiers_minDCF(DTR, LTR, prior=0.5, cost_fn=1, cost_fp=1, k=k, eval_data=eval_data)
    print()
    print("********************************************************************")
    '''
    

    
    ### -- LINEAR LOGISTIC REGRESSION
    '''
    print("********************* LR TABLE ************************************")
    print("------> applicazione prior = 0.5")
    print_table_LR_minDCF(DTR,LTR, prior=0.5, cost_fn=1, cost_fp=1, k=k, eval_data=eval_data)
    print()
    print("********************************************************************")
    '''
    
    ### -- QUADRATIC LOGISTIC REGRESSION
    
    
    print("********************* QUADRATIC LR TABLE ************************************")
    print("------> applicazione prior = 0.5")
    print_table_Quadratic_LR_minDCF(DTR,LTR, prior=0.5, cost_fn=1, cost_fp=1, k=k, eval_data=eval_data)
    print()
    print("********************************************************************")
    
    ### -- LINEAR SVM
    '''
    print("********************* SVM TABLES ************************************")
    print("------> applicazione con prior = 0.5")
    print_table_SVM_minDCF(DTR, LTR, prior=0.5, cost_fn=1, cost_fp=1, k=k, eval_data=eval_data)
    print("********************************************************************")
    '''
    
    ### -- QUADRATIC SVM
    '''
    print("********************* quadratic SVM TABLES ************************************")
    print("------> applicazione con prior = 0.5")
    print_table_Quadratic_SVM_minDCF(DTR, LTR, prior=0.5, cost_fn=1, cost_fp=1, k=k, eval_data=eval_data)
    print("********************************************************************")
    '''
    
    ### RBF
    '''
    print("********************* RBF SVM TABLES ************************************")
    print("------> applicazione con prior = 0.5")
    print_table_RBF_SVM_minDCF(DTR, LTR, prior=0.5, cost_fn=1, cost_fp=1, k=k, eval_data=eval_data)    
    print("********************************************************************")
    '''
    
    ### GMM
    '''
    print_table_GMM_minDCF(DTR, LTR, k, eval_data = eval_data)
    '''



#--------------------------
