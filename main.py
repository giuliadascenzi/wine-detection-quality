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
        min_DCF_Tied = model_evaluation.k_cross_minDCF(data, LTR,k, MVGclassifiers.TIED_logLikelihoodRatios, prior , cost_fn, cost_fp)
        print("[5- Fold] - Tied MVG: ",min_DCF_Tied)

        #Tied Diag_Cov
        min_DCF_Tied_Diag_Cov = model_evaluation.singleFold_minDCF(data, LTR, MVGclassifiers.TIED_DIAG_COV_logLikelihoodRatios, prior , cost_fn, cost_fp)
        print("[Single Fold] - Tied MVG with Diag Cov: ",min_DCF_Tied_Diag_Cov)
        min_DCF_Tied_Diag_Cov = model_evaluation.k_cross_minDCF(data, LTR, k,  MVGclassifiers.TIED_DIAG_COV_logLikelihoodRatios, prior , cost_fn, cost_fp)
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
    gaussianizedFeatures = gaussianization(DTR)

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

def print_table_LR_minDCF(DTR, LTR, prior, cost_fn, cost_fp, k):

    def LR_minDCF(data):
            lam = 10**(-7)
            pi_T = 0.5
            min_DCF_LR = model_evaluation.k_cross_minDCF(data, LTR, k, logisticRegression.LR_logLikelihoodRatios, prior , cost_fn, cost_fp, [lam, pi_T])
            print("[5-Folds]  -  lam = 10^-7, pi_T = 0.5: ",min_DCF_LR)  

            pi_T = 0.1
            min_DCF_LR = model_evaluation.k_cross_minDCF(data, LTR, k, logisticRegression.LR_logLikelihoodRatios, prior , cost_fn, cost_fp, [lam, pi_T])
            print("[5-Folds]  -  lam = 10^-7, pi_T = 0.1: ",min_DCF_LR)

            pi_T = 0.9
            min_DCF_LR = model_evaluation.k_cross_minDCF(data, LTR, k, logisticRegression.LR_logLikelihoodRatios, prior , cost_fn, cost_fp, [lam, pi_T])
            print("[5-Folds]  -  lam = 10^-7, pi_T = 0.9: ",min_DCF_LR)

            print()

    
    #------------------------RAW FEATURES -----------------
    print("*** minDCF - RAW FEATURES ***")
    LR_minDCF(DTR)

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
            minDCFs[i] = model_evaluation.singleFold_minDCF(data, LTR, logisticRegression.LR_logLikelihoodRatios, prior , cost_fn, cost_fp, [lam, pi_T])
        
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
            minDCFs[i] = model_evaluation.k_cross_minDCF(data, LTR,k, logisticRegression.LR_logLikelihoodRatios, prior , cost_fn, cost_fp, [lam, pi_T])
        
        lb = "minDCF (prior="+ str(prior)+ " )"
        print("DONE")
        plt.plot(lambdas, minDCFs, label=lb)
        plt.legend()

    plt.figure()
    print("+++++++++++++++++++++++++++++++++++")
    print("Raw fearures, single fold")
    plt.title("Raw fearures, single fold")
    plt.xscale('log')
    plt.xlabel("lambda")
    plt.ylabel("minDCFs")
    oneGraphSingleFold(DTR, prior=0.5, cost_fn=1, cost_fp=1, pi_T=0.5)
    oneGraphSingleFold(DTR, prior=0.1, cost_fn=1, cost_fp=1, pi_T=0.5)
    oneGraphSingleFold(DTR, prior=0.9, cost_fn=1, cost_fp=1, pi_T=0.5)
    plt.savefig('Graph/LR/singleFoldRAW.png' )

    plt.figure()
    print("+++++++++++++++++++++++++++++++++++")
    print("Gaussianized fearures, single fold")
    plt.title("Gaussianized fearures, single fold")
    plt.xscale('log')
    plt.xlabel("lambda")
    plt.ylabel("minDCFs")
    gaussianizedFeatures = gaussianization(DTR)
    oneGraphSingleFold(gaussianizedFeatures, prior=0.5, cost_fn=1, cost_fp=1, pi_T=0.5)
    oneGraphSingleFold(gaussianizedFeatures, prior=0.1, cost_fn=1, cost_fp=1, pi_T=0.5)
    oneGraphSingleFold(gaussianizedFeatures, prior=0.9, cost_fn=1, cost_fp=1, pi_T=0.5)
    plt.savefig('Graph/LR/singleFoldGauss.png' )

    plt.figure()
    print("+++++++++++++++++++++++++++++++++++")
    print("Raw fearures, 5 fold")
    plt.title("Raw fearures, 5 fold")
    plt.xscale('log')
    plt.xlabel("lambda")
    plt.ylabel("minDCFs")
    oneGraphKFold(DTR, prior=0.5, cost_fn=1, cost_fp=1, pi_T=0.5)
    oneGraphKFold(DTR, prior=0.1, cost_fn=1, cost_fp=1, pi_T=0.5)
    oneGraphKFold(DTR, prior=0.9, cost_fn=1, cost_fp=1, pi_T=0.5)
    plt.savefig('Graph/LR/5FoldRAW.png' )

    plt.figure()
    print("+++++++++++++++++++++++++++++++++++")
    print("Gaussianized fearures, 5 fold")
    plt.title("Gaussianized fearures, 5 fold")
    plt.xscale('log')
    plt.xlabel("lambda")
    plt.ylabel("minDCFs")
    gaussianizedFeatures = gaussianization(DTR)
    oneGraphKFold(gaussianizedFeatures, prior=0.5, cost_fn=1, cost_fp=1, pi_T=0.5)
    oneGraphKFold(gaussianizedFeatures, prior=0.1, cost_fn=1, cost_fp=1, pi_T=0.5)
    oneGraphKFold(gaussianizedFeatures, prior=0.9, cost_fn=1, cost_fp=1, pi_T=0.5)
    plt.savefig('Graph/LR/5FoldGauss.png' )
    #plt.show()

    
 


def print_graphs_SVM_Cs(DTR, LTR, k ):

    def oneGraphSingleFold(data, prior, cost_fn, cost_fp, pi_T):
        print("working on single fold prior = ", prior)
        exps = numpy.linspace(-3,1, 5)
        Cs = 10** exps
        minDCFs = 0 * exps
        for i in range (Cs.size):
            C= Cs[i]
            minDCFs[i] = model_evaluation.singleFold_minDCF(data, LTR, SVMClassifier.SVM_computeLogLikelihoods, prior , cost_fn, cost_fp, [pi_T, C])
        
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
            minDCFs[i] = model_evaluation.k_cross_minDCF(data, LTR,k, SVMClassifier.SVM_computeLogLikelihoods, prior , cost_fn, cost_fp, [pi_T, C])
        
        lb = "minDCF (prior="+ str(prior) +")"
        plt.plot(Cs, minDCFs, label=lb)
        plt.legend()
        print("DONE")

    plt.figure()
    print("1 grafico")
    plt.title("Raw fearures, single fold")
    plt.xscale('log')
    plt.xlabel("C")
    plt.ylabel("minDCFs")
    oneGraphSingleFold(DTR, prior=0.5, cost_fn=1, cost_fp=1, pi_T=0.5)
    oneGraphSingleFold(DTR, prior=0.1, cost_fn=1, cost_fp=1, pi_T=0.5)
    oneGraphSingleFold(DTR, prior=0.9, cost_fn=1, cost_fp=1, pi_T=0.5)
    plt.savefig('Graph/SVM/linear/singleFoldRAW.png' )
    
    print("2 grafico")
    plt.figure()
    plt.title("Gaussianized fearures, single fold")
    plt.xscale('log')
    plt.xlabel("C")
    plt.ylabel("minDCFs")
    gaussianizedFeatures = gaussianization(DTR)
    oneGraphSingleFold(gaussianizedFeatures, prior=0.5, cost_fn=1, cost_fp=1, pi_T=0.5)
    oneGraphSingleFold(gaussianizedFeatures, prior=0.1, cost_fn=1, cost_fp=1, pi_T=0.5)
    oneGraphSingleFold(gaussianizedFeatures, prior=0.9, cost_fn=1, cost_fp=1, pi_T=0.5)
    plt.savefig('Graph/SVM/linear/singleFoldGauss.png' )
   
    print("3 grafico")
    plt.figure()
    plt.title("Raw fearures, 5 fold")
    plt.xscale('log')
    plt.xlabel("C")
    plt.ylabel("minDCFs")
    oneGraphKFold(DTR, prior=0.5, cost_fn=1, cost_fp=1, pi_T=0.5)
    oneGraphKFold(DTR, prior=0.1, cost_fn=1, cost_fp=1, pi_T=0.5)
    oneGraphKFold(DTR, prior=0.9, cost_fn=1, cost_fp=1, pi_T=0.5)
    plt.savefig('Graph/SVM/linear/5FoldRAW.png' )

    print("4 grafico")
    plt.figure()
    plt.title("Gaussianized fearures, 5 fold")
    plt.xscale('log')
    plt.xlabel("C")
    plt.ylabel("minDCFs")
    gaussianizedFeatures = gaussianization(DTR)
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
            minDCFs[i] = model_evaluation.singleFold_minDCF(data, LTR, SVMClassifier.Polinomial_SVM_computeLogLikelihoods, prior , cost_fn, cost_fp, [pi_T, C])
        
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
            minDCFs[i] = model_evaluation.k_cross_minDCF(data, LTR,k, SVMClassifier.Polinomial_SVM_computeLogLikelihoods, prior , cost_fn, cost_fp, [pi_T, C])
        
        lb = "minDCF (prior="+ str(prior) +")"
        plt.plot(Cs, minDCFs, label=lb)
        plt.legend()
        print("DONE")

    plt.figure()
    print("1 grafico")
    plt.title("Raw fearures, single fold")
    plt.xscale('log')
    plt.xlabel("C")
    plt.ylabel("minDCFs")
    oneGraphSingleFold(DTR, prior=0.5, cost_fn=1, cost_fp=1, pi_T=0.5)
    oneGraphSingleFold(DTR, prior=0.1, cost_fn=1, cost_fp=1, pi_T=0.5)
    oneGraphSingleFold(DTR, prior=0.9, cost_fn=1, cost_fp=1, pi_T=0.5)
    plt.savefig('Graph/SVM/Quadratic/singleFoldRAW.png' )
    
    print("2 grafico")
    plt.figure()
    plt.title("Gaussianized fearures, single fold")
    plt.xscale('log')
    plt.xlabel("C")
    plt.ylabel("minDCFs")
    gaussianizedFeatures = gaussianization(DTR)
    oneGraphSingleFold(gaussianizedFeatures, prior=0.5, cost_fn=1, cost_fp=1, pi_T=0.5)
    oneGraphSingleFold(gaussianizedFeatures, prior=0.1, cost_fn=1, cost_fp=1, pi_T=0.5)
    oneGraphSingleFold(gaussianizedFeatures, prior=0.9, cost_fn=1, cost_fp=1, pi_T=0.5)
    plt.savefig('Graph/SVM/Quadratic/singleFoldGauss.png' )
   
    print("3 grafico")
    plt.figure()
    plt.title("Raw fearures, 5 fold")
    plt.xscale('log')
    plt.xlabel("C")
    plt.ylabel("minDCFs")
    oneGraphKFold(DTR, prior=0.5, cost_fn=1, cost_fp=1, pi_T=0.5)
    oneGraphKFold(DTR, prior=0.1, cost_fn=1, cost_fp=1, pi_T=0.5)
    oneGraphKFold(DTR, prior=0.9, cost_fn=1, cost_fp=1, pi_T=0.5)
    plt.savefig('Graph/SVM/Quadratic/5FoldRAW.png' )

    print("4 grafico")
    plt.figure()
    plt.title("Gaussianized fearures, 5 fold")
    plt.xscale('log')
    plt.xlabel("C")
    plt.ylabel("minDCFs")
    gaussianizedFeatures = gaussianization(DTR)
    oneGraphKFold(gaussianizedFeatures, prior=0.5, cost_fn=1, cost_fp=1, pi_T=0.5)
    oneGraphKFold(gaussianizedFeatures, prior=0.1, cost_fn=1, cost_fp=1, pi_T=0.5)
    oneGraphKFold(gaussianizedFeatures, prior=0.9, cost_fn=1, cost_fp=1, pi_T=0.5)
    plt.savefig('Graph/SVM/Quadratic/5FoldGauss.png' )
    #plt.show()



def print_graphs_RBF_SVM_Cs(DTR, LTR, k):

    def oneGraphSingleFold(data, prior, cost_fn, cost_fp, pi_T, lam):
        print("working on single fold prior = ", prior)
        exps = numpy.linspace(-3,1, 5)
        Cs = 10** exps
        minDCFs = 0 * exps
        for i in range (Cs.size):
            C= Cs[i]
            minDCFs[i] = model_evaluation.singleFold_minDCF(data, LTR, SVMClassifier.RBF_SVM_computeLogLikelihoods, prior , cost_fn, cost_fp, [pi_T, C, lam])
        
        lb = "minDCF (prior="+ str(prior) +")"
        plt.plot(Cs, minDCFs, label=lb)
        plt.legend()
        print("DONE")

    def oneGraphKFold(data, prior, cost_fn, cost_fp, pi_T, lam):
        print("working on k fold prior = ", prior)
        exps = numpy.linspace(-3,1, 5)
        Cs = 10** exps
        minDCFs = 0 * exps
        for i in range (Cs.size):
            C= Cs[i]
            minDCFs[i] = model_evaluation.k_cross_minDCF(data, LTR,k, SVMClassifier.RBF_SVM_computeLogLikelihoods, prior , cost_fn, cost_fp, [pi_T, C, lam])
        
        lb = " (log(lam)="+ str(numpy.log(lam)) +")"
        plt.plot(Cs, minDCFs, label=lb)
        plt.legend()
        print("DONE")

    


    print("1 grafico")
    plt.figure()
    plt.title("Raw fearures, 5 fold")
    plt.xscale('log')
    plt.xlabel("C")
    plt.ylabel("minDCFs")
    oneGraphKFold(DTR, prior=0.5, cost_fn=1, cost_fp=1, pi_T=0.5, lam = 10**0)
    oneGraphKFold(DTR, prior=0.5, cost_fn=1, cost_fp=1, pi_T=0.5, lam = 10**-1)
    oneGraphKFold(DTR, prior=0.5, cost_fn=1, cost_fp=1, pi_T=0.5, lam = 10**-2)
    oneGraphKFold(DTR, prior=0.5, cost_fn=1, cost_fp=1, pi_T=0.5, lam = 10**-2)
    plt.savefig('Graph/SVM/RBF/5FoldRAW.png' )

    print("2 grafico")
    plt.figure()
    plt.title("Gaussianized fearures, 5 fold")
    plt.xscale('log')
    plt.xlabel("C")
    plt.ylabel("minDCFs")
    gaussianizedFeatures = gaussianization(DTR)
    oneGraphKFold(gaussianizedFeatures, prior=0.5, cost_fn=1, cost_fp=1, pi_T=0.5, lam = 10**0)
    oneGraphKFold(gaussianizedFeatures, prior=0.5, cost_fn=1, cost_fp=1, pi_T=0.5, lam = 10**-1)
    oneGraphKFold(gaussianizedFeatures, prior=0.5, cost_fn=1, cost_fp=1, pi_T=0.5, lam = 10**-2)
    oneGraphKFold(gaussianizedFeatures, prior=0.5, cost_fn=1, cost_fp=1, pi_T=0.5, lam = 10**-3)
    plt.savefig('Graph/SVM/RBF/5FoldGauss.png' )
    #plt.show()


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
    #gaussianizedFeatures = gaussianization(DTR)
    #stats.plot_hist(gaussianizedFeatures, LTR)


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

    print("********************* SVM GRAPHS ************************************")
    print_graphs_SVM_Cs(DTR, LTR, k=k )
    print("********************************************************************")

    '''
    '''
    print("********************* quadratic SVM GRAPHS ************************************")
    print_graphs_Polinomial_SVM_Cs(DTR, LTR, k=k )
    print("********************************************************************")

    '''

    print("********************* RBF SVM GRAPHS ************************************")
    print_graphs_RBF_SVM_Cs(DTR, LTR, k=k )
    print("********************************************************************")



    
    #stats.plot_scatter(principal_components,LTR)
    #linear_discriminants = redTec.LDA(DTR,LTR, 1)
    #redTec.plotLDA(linear_discriminants, LTR, "Applied LDA")


   

          