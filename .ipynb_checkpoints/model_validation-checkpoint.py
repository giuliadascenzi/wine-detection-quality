import numpy
import matplotlib.pyplot as plt
from numpy.random import permutation


def mcol(v):
    return v.reshape((v.size, 1))
def mrow(v):
    return v.reshape((1, v.size))

def split_db_2tol(D,L, seed=0):
    nTrain=int(D.shape[1]*2.0/3.0)
    numpy.random.seed(seed)
    idx=numpy.random.permutation(D.shape[1])
    idxTrain=idx[0:nTrain]
    idxTest=idx[nTrain:]

    #Training Data
    DTR = D[:, idxTrain]
    #Evaluation Data
    DTE = D[:, idxTest]
    #Training Labels
    LTR = L[idxTrain]
    #Evaluation Labels
    LTE = L[idxTest]

    return [(DTR, LTR), (DTE,LTE)]

def compute_confusion_matrix(predicted_labels, actual_labels, numClasses):
    #Build confusion matrix 
    c_matrix_C =numpy.zeros((numClasses,numClasses))
    #columns =classes, #rows= predictions

    # classLabels: evaluation labels -> actual class labels
    # predicted_labelsC ->assigned class Labels    
    for i in range (len(actual_labels)):
        columnIndex=actual_labels[i]
        rowIndex=predicted_labels[i]
        c_matrix_C[rowIndex][columnIndex]+=1
    return c_matrix_C



def compute_optimal_bayes_decision(loglikelihood_ratios, prior, cost_fn, cost_fp):
    threshold= - numpy.log((prior*cost_fn)/((1-prior)*cost_fp))
    return(1*(loglikelihood_ratios>threshold))


def compute_FNR(conf_matrix):
    return conf_matrix[0][1] /(conf_matrix[0][1] + conf_matrix[1][1] )

def compute_FPR(conf_matrix):
    return  conf_matrix[1][0] /(conf_matrix[0][0] + conf_matrix[1][0] )

def compute_bayes_risk(conf_matrix, prior, cost_fn, cost_fp):
    FNR= compute_FNR(conf_matrix)
    FPR= compute_FPR(conf_matrix)

    risk= prior*cost_fn*FNR+(1-prior)*cost_fp*FPR
    return risk

def compute_normalized_bayes_risk(bayes_risk ,prior, cost_fn, cost_fp):
    return bayes_risk/(min(prior*cost_fn, (1-prior)*cost_fp))



def compute_minimum_detection_cost(llrs, labels, prior, cost_fn, cost_fp):
    # 1) ordina in ordine crescente i test scores= data (logLikelihood ratios)
    llrs_sorted= numpy.sort(llrs)
    # 2) considero ogni elemento data come threshold, ottengo le predicted labels confrontando con la threshold
    DCFs=[]
    FPRs=[]
    TPRs=[]

    for t in llrs_sorted:
        p_label=1*(llrs>t)
        conf_matrix=compute_confusion_matrix(p_label, labels, numpy.unique(labels).size )
        br= compute_bayes_risk(conf_matrix, prior, cost_fn, cost_fp)
        nbr= compute_normalized_bayes_risk(br, prior, cost_fn, cost_fp)
        DCFs.append(nbr)

        #salvare nei dati della roc FPR e TPR
        FPRs.append(compute_FPR(conf_matrix))
        TPRs.append(1-compute_FNR(conf_matrix))
    
    DCF_min =min(DCFs)

    index_t = DCFs.index(DCF_min)
    
    return (DCF_min, FPRs, TPRs, llrs_sorted[index_t])

def compute_actual_DCF(llrs, labels, prior , cost_fn, cost_fp):
    #predicted labels using the theoretical threshold
    p_label=compute_optimal_bayes_decision(llrs, prior, cost_fn, cost_fp)
    #build confusion matrix
    conf_matrix=compute_confusion_matrix(p_label, labels, numpy.unique(labels).size )
    #bayes risk
    br= compute_bayes_risk(conf_matrix, prior, cost_fn, cost_fp)
    # normalized bayes risk -> actual DCF
    nbr= compute_normalized_bayes_risk(br, prior, cost_fn, cost_fp)

   
    return (nbr)



def k_cross_loglikelihoods(D,L, k, llr_calculator, otherParams):
    step = int(D.shape[1]/k)
    numpy.random.seed(seed=0)

    random_indexes = numpy.random.permutation(D.shape[1])

    llr = []
    labels = []

    for i in range(k):
        if i == k-1:
            indexesEV = random_indexes[i*step:]
            indexesTR = random_indexes[:i*step]
            
        elif i==0:
            indexesEV = random_indexes[0:step]
            indexesTR = random_indexes[step:]

        else:
            indexesEV = random_indexes[i*step:(i+1)*step]
            tmp1 = random_indexes[: i*step]
            tmp2 = random_indexes[(i+1)*step:]
            indexesTR = numpy.concatenate((tmp1,tmp2), axis=None)

        DTR = D[:, indexesTR]
        LTR = L[indexesTR]

        DEV = D[:, indexesEV]
        LEV = L[indexesEV]
        
        llr_i= llr_calculator(DTR, LTR, DEV, otherParams)
        llr.append(llr_i)
        labels.append(LEV)

    llr = numpy.concatenate(llr)
    labels = numpy.concatenate(labels)
    return (llr, labels)

def k_cross_DCF(D, L, k, llr_calculator, prior, cost_fn, cost_fp, otherParams=None, eval_data=None):
    if (eval_data!=None): 
        DEV = eval_data[0]
        labels= eval_data[1]
        llr = llr_calculator(D, L, DEV, otherParams)
    else :
        llr, labels = k_cross_loglikelihoods(D,L,k, llr_calculator, otherParams)

    actDCF = compute_actual_DCF(llr, labels, prior , cost_fn, cost_fp)
    min_DCF,_,_,optimal_treshold =compute_minimum_detection_cost(llr, labels, prior , cost_fn, cost_fp)
    return (min_DCF, actDCF, optimal_treshold)


def singleFold_DCF(D, L, llr_calculator, prior, cost_fn, cost_fp, otherParams=None,  eval_data=None): #eval_data=[DTE,LTE]
    (DTR, LTR), (DTE,LTE)= split_db_2tol(D,L)
    if (eval_data!=None):
        DTE = eval_data[0]
        LTE= eval_data[1]

    llr = llr_calculator (DTR,LTR,DTE, otherParams)
    actDCF = compute_actual_DCF(llr, LTE, prior , cost_fn, cost_fp)
    min_DCF,_,_,optimal_treshold =compute_minimum_detection_cost(llr, LTE, prior , cost_fn, cost_fp)
    return (min_DCF, actDCF, optimal_treshold) #minDCF

def bayes_error_plot(D, L, k, llr_calculator, otherParams, title, color ):

    llr, labels = k_cross_loglikelihoods(D,L, k, llr_calculator, otherParams)

    effPriorLogOdds = numpy.linspace(-3,3,21)
    effPriors = 1 / (1+ numpy.exp(-effPriorLogOdds))
    dcf = []
    mindcf = []

    for effPrior in effPriors:
        #calculate actual dcf considering effPrior
        d = compute_actual_DCF(llr, labels, effPrior , 1, 1)
        #calculate min dcf considering effPrior
        m,_,_,_ =compute_minimum_detection_cost(llr, labels, effPrior , 1, 1)
        dcf.append(d)
        mindcf.append(m)
    

    plt.plot(effPriorLogOdds, dcf, color ,label=title+' DCF')
    plt.plot(effPriorLogOdds, mindcf, color+"--", label=title+ ' min DCF')
    plt.ylim([0,1.1])
    plt.xlim([-3,3])
    plt.legend()
    

def compute_DCF_with_optimal_treshold(D, L, k, llr_calculator, otherParams, prior, cost_fn, cost_fp ):
                
    #1st: calculate the loglikelihood ratios using k-cross method
    llr, labels = k_cross_loglikelihoods(D, L, k, llr_calculator, otherParams)
    
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
    minDCF,_,_,optimal_treshold = compute_minimum_detection_cost(llr1, labels1, prior , cost_fn, cost_fp)

    predicted_labels = 1*(llr2 > optimal_treshold)

    conf_matrix=compute_confusion_matrix(predicted_labels, labels2, numpy.unique(labels2).size )
    br= compute_bayes_risk(conf_matrix, prior, cost_fn, cost_fp)

    #nbr is the DCF obtained with the estimated optimal treshold
    nbr= compute_normalized_bayes_risk(br, prior, cost_fn, cost_fp)
    
    #actual DCF done with theoretical optimal treshold
    actDCF = compute_actual_DCF(llr2, labels2, prior , cost_fn, cost_fp)

    #minDCF 
    minDCF,_,_,_ = compute_minimum_detection_cost(llr2, labels2, prior , cost_fn, cost_fp)


    return (minDCF, actDCF, nbr, optimal_treshold)