import numpy


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





def compute_normalised_bayes_risk_wrapper(data, labels, prior, cost_fn, cost_fp):

    #return 1 if a sample is greater than the trashold, 0 otherwise
    predicted_labels = compute_optimal_bayes_decision(data, prior, cost_fn, cost_fp)

    #Build the corresponding confusion matrix 
    c_matrix_IP =compute_confusion_matrix(predicted_labels, labels, 2 )
    bayes_risk= compute_bayes_risk(c_matrix_IP,prior, cost_fn, cost_fp)
    normalized_bayes_risk=compute_normalized_bayes_risk(bayes_risk ,prior, cost_fn, cost_fp)
    return (predicted_labels,c_matrix_IP, bayes_risk, normalized_bayes_risk)

def compute_minimum_detection_cost(data, labels, prior, cost_fn, cost_fp):
    # 1) ordina in ordine crescente i test scores= data (logLikelihood ratios)
    data_sorted= numpy.sort(data)
    # 2) considero ogni elemento data come threshold, ottengo le predicted labels confrontando con la threshold
    DCFs=[]
    FPRs=[]
    TPRs=[]

    for t in data_sorted:
        p_label=1*(data>t)
        conf_matrix=compute_confusion_matrix(p_label, labels, numpy.unique(labels).size )
        br= compute_bayes_risk(conf_matrix, prior, cost_fn, cost_fp)
        nbr= compute_normalized_bayes_risk(br, prior, cost_fn, cost_fp)
        DCFs.append(nbr)

        #salvare nei dati della roc FPR e TPR
        FPRs.append(compute_FPR(conf_matrix))
        TPRs.append(1-compute_FNR(conf_matrix))
    
    DCF_min =min(DCFs)
    
    return (DCF_min, FPRs, TPRs)


def k_cross_minDCF(D, L, k, llr_calculator, prior , cost_fn, cost_fp):
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
        
        llr_i= llr_calculator(DTR, LTR, DEV)
        llr.append(llr_i)
        labels.append( LEV)

    llr = numpy.concatenate(llr)
    labels = numpy.concatenate(labels)
    min_DCF,_,_ =compute_minimum_detection_cost(llr, labels, prior , cost_fn, cost_fp)
    return min_DCF #minDCF

