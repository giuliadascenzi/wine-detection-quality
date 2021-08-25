import numpy

def mcol(v):
    return v.reshape((v.size, 1))

def means(samples, labels): #one time for each class
    means = []

    for i in range (numpy.unique(labels).size):
        class_samples =samples[:, labels==i] # results matrix of 4 rows and # columns (#= number of data in that class)
        means.append(class_samples.mean(1))  # compute the mean of the columns 
    return means #1-D array 



def covariances(samples, labels, means):
    covariances= []

    for i in range (numpy.unique(labels).size): #one time for each class
        class_samples=samples[:, labels==i]
        # to compute the covariance matrix in a efficiently way:
        # 1) center the data removing the mean from all points (the mean in this case is a 1-D array so I need to convert it in a column array)
        centered_samples=class_samples - mcol(means[i]) 
        covariance_matrix=numpy.dot(centered_samples, centered_samples.T) / centered_samples.shape[1]
        covariances.append(covariance_matrix)

    return covariances


def compute_loglikelihood(sample, mu, sigma): #num variables= number of features! in this case 4
    M = sample.shape[0] # number of features! in this case 4
    a = (-M/2) * numpy.log(2*numpy.pi)
    b = (-0.5) * numpy.log( numpy.linalg.det(sigma) )
    
    norma = sample-mu
    sigma_inv = numpy.linalg.inv(sigma)
    
    c=numpy.dot(sigma_inv, norma)
    c = -0.5 *numpy.dot(norma.T, c)
    res = a+b+c
    return res


def compute_likelihoods(samples, means, covariances, numlabels):
    S= numpy.zeros((numlabels, samples.shape[1])) #score result matrix= matrix  #Row=classes x #columns=data_sample
    
    #for each semple compute the likelihood for every class
    for nClass in range (numlabels):

        for  j in range (samples.shape[1]):

            sample = samples[:, j]
            mean =means[nClass]
            covariance =covariances[nClass]
            loglikelihood =compute_loglikelihood(sample, mean, covariance)
            S[nClass][j] = numpy.exp(loglikelihood)
    return S



def compute_loglikelihoods(samples, means, covariances, numlabels):
    S= numpy.zeros((numlabels, samples.shape[1])) #score result matrix= matrix  #Row=classes x #columns=data_sample
    
    #for each semple compute the likelihood for every class
    for nClass in range (numlabels):

        for  j in range (samples.shape[1]):

            sample = samples[:, j]
            mean =means[nClass]
            covariance =covariances[nClass]
            loglikelihood = compute_loglikelihood(sample, mean, covariance)
            S[nClass][j] = loglikelihood
    return S



def compute_joint_log_distribution (scores, classes_prior_probabilties):
    #scores: matrix #row= #classes, #columns=#data sample

    S= numpy.zeros((scores.shape[0], scores.shape[1])) #score result matrix= matrix  #Row=classes x #columns=data_sample
    
    for nClass in range (scores.shape[0]):
        S[nClass, :]= scores[nClass, :]+ numpy.log(classes_prior_probabilties[nClass])

    return S



def compute_log_class_posterior_probabilities (joint_scores):
    from scipy import special
    marginal_logdensity = special.logsumexp(joint_scores, axis=0)
    log_post = joint_scores - marginal_logdensity

    return log_post



def compute_joint_distribution (scores, classes_prior_probabilties):
    #scores: matrix #row= #classes, #columns=#data sample

    S= numpy.zeros((scores.shape[0], scores.shape[1])) #score result matrix= matrix  #Row=classes x #columns=data_sample
    
    for nClass in range (scores.shape[0]):
        S[nClass, :]= scores[nClass, :]*classes_prior_probabilties[nClass]

    
    return S





def compute_class_posterior_probabilities (joint_scores):

    S= numpy.zeros((joint_scores.shape[0], joint_scores.shape[1])) #score result array #num_samples
    
    for nSample in range (joint_scores.shape[1]):
        
        denominator = joint_scores[: , nSample].sum()

        for nClass in range (joint_scores.shape[0]):
           
            num = joint_scores[nClass][nSample]
            S[nClass][nSample]= num/denominator
    
    return S




def compute_accuracy_error(predicted_labels, LTE):
    good_predictions = (predicted_labels == LTE) #array with True when predicted_labels[i] == LTE[i]    
    num_corrected_predictions =(good_predictions==True).sum()
    tot_predictions = predicted_labels.shape[0]
    accuracy= num_corrected_predictions /tot_predictions
    error = (tot_predictions - num_corrected_predictions ) /tot_predictions

    return (accuracy, error)




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
