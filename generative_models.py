import numpy

def mcol(v):
    return v.reshape((v.size, 1))
def mrow(v):
    return v.reshape((1, v.size))

def MVG_classifier(DTR, LTR, DTE, LTE): #TODO : passare prior probability come parametro (anche negli altri classificatori)
    n_classes = numpy.unique(LTR).size
    # 1) Compute the ML estimetes for the classifier parameters (mu, co : for each class) mu=mean, co=covariance matrix
    m = means(DTR, LTR)
    c = covariances(DTR, LTR, m)

    P_c = 1/ n_classes
    predicted_labels = compute_MVG_classification(DTE, n_classes, m, c, P_c)
    # 6) compute accuracy and error rate
    accuracy, error = compute_accuracy_error(predicted_labels, LTE)

    return (accuracy*predicted_labels.shape[0]) #return tot_true_predictions


def NaiveBayes_classifier(DTR, LTR, DTE, LTE):
    # 1) Compute the ML estimetes for the classifier parameters (mu, co : for each class) mu=mean, co=covariance matrix
    m = means(DTR, LTR)
    c_MVG =covariances(DTR, LTR, m)
    n_classes = numpy.unique(LTR).size


    # get the diagonal only from the matrix
    c = c_MVG
    for i in range (len(c_MVG)):
        c[i]= c_MVG[i]*numpy.eye(c_MVG[i].shape[0],c_MVG[i].shape[1])

    P_c = 1/ n_classes
    predicted_labels = compute_MVG_classification(DTE, n_classes, m, c, P_c)
    # 6) compute accuracy and error rate
    accuracy, error = compute_accuracy_error(predicted_labels, LTE)

    return (accuracy*predicted_labels.shape[0]) #return tot_true_predictions


def TiedCovariance_classifier(DTR, LTR, DTE, LTE):
     # 1) Compute the ML estimetes for the classifier parameters (mu, co : for each class) mu=mean, co=covariance matrix
    m = means(DTR, LTR)
    c_MVG =covariances(DTR, LTR, m)
    n_classes = numpy.unique(LTR).size

    # 2) get the tied covariance
    csigma = numpy.zeros((DTR.shape[0], DTR.shape[0]))

    for i in range (len(c_MVG)):
        Nc = (LTR == i).sum()
        csigma += c_MVG[i]*Nc
    csigma = csigma / LTR.size

    c =[]
    for i in range (len(c_MVG)):
        c.append(csigma)

    P_c = 1/ n_classes
    predicted_labels = compute_MVG_classification(DTE, n_classes, m, c, P_c)
    # 6) compute accuracy and error rate
    accuracy, error = compute_accuracy_error(predicted_labels, LTE) 

    return (accuracy*predicted_labels.shape[0]) #return tot_true_predictions

def TiedDiagCovariance_classifier(DTR, LTR, DTE, LTE):
     # 1) Compute the ML estimetes for the classifier parameters (mu, co : for each class) mu=mean, co=covariance matrix
    m = means(DTR, LTR)
    c_MVG =covariances(DTR, LTR, m)
    n_classes = numpy.unique(LTR).size

    # 2) get the tied covariance
    csigma = numpy.zeros((DTR.shape[0], DTR.shape[0]))

    for i in range (len(c_MVG)):
        Nc = (LTR == i).sum()
        csigma += c_MVG[i]*numpy.eye(c_MVG[i].shape[0],c_MVG[i].shape[1])*Nc
    csigma = csigma / LTR.size

    c =[]
    for i in range (len(c_MVG)):
        c.append(csigma)

    P_c = 1/ n_classes
    predicted_labels = compute_MVG_classification(DTE, n_classes, m, c, P_c)
    # 6) compute accuracy and error rate
    accuracy, error = compute_accuracy_error(predicted_labels, LTE) 

    return (accuracy*predicted_labels.shape[0]) #return tot_true_predictions

    
def compute_MVG_classification(DTE, n_classes, means, covariances, P_c):
    # 2) Compute the likelihood of each sample for each class and store it in a matrix #Row=classes x #columns=data_sample
    scores = compute_likelihoods(DTE, means, covariances, n_classes)
    # 3) Compute class posterior probability= multiply the score matrix with prior probability
    classes_prior_probabilities = numpy.array([P_c, P_c, P_c]) #assuming for each class the same P(c)
    S_joint =compute_joint_distribution(scores, classes_prior_probabilities) 
    # 4) compute class posterior probability
    S_post =compute_class_posterior_probabilities(S_joint) 
        

    # 5) predicted labels= class that has the maximum posterior probabiliti
    predicted_labels = S_post.argmax(0) #argmax over the rows = for each column selects the index of the row with the highes value
    return predicted_labels


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

def compute_likelihoods(samples, means, covariances, numlabels):
    S= numpy.zeros((numlabels, samples.shape[1])) #score result matrix= matrix  #Row=classes x #columns=data_sample
    
    #for each semple compute the likelihood for every class
    for nClass in range (numlabels):
        for  j in range (samples.shape[1]):
            sample = samples[:, j]
            mean =means[nClass]
            covariance =covariances[nClass]
            loglikelihood =compute_MVG_logdensity(sample, mean, covariance)
            S[nClass][j] = numpy.exp(loglikelihood)
    return S

def compute_MVG_logdensity(sample, mu, sigma): #num variables= number of features! in this case 4
    M = sample.shape[0] # number of features! in this case 4
    a = (-M/2) * numpy.log(2*numpy.pi)
    b = (-0.5) * numpy.log( numpy.linalg.det(sigma) )
    
    norma = sample-mu
    sigma_inv = numpy.linalg.inv(sigma)
    
    c = -0.5 *numpy.linalg.multi_dot([norma.T, sigma_inv, norma])
    res = a+b+c
    return res

def compute_joint_distribution (scores, classes_prior_probabilities):
    #scores: matrix #row= #classes, #columns=#data sample
    
    return (scores*mcol(classes_prior_probabilities))

def compute_class_posterior_probabilities (joint_scores):
    return joint_scores / mrow(joint_scores.sum(0))

def compute_accuracy_error(predicted_labels, LTE):
    good_predictions = (predicted_labels == LTE) #array with True when predicted_labels[i] == LTE[i]    
    num_corrected_predictions =(good_predictions==True).sum()
    tot_predictions = predicted_labels.size
    accuracy= num_corrected_predictions /tot_predictions
    error = (tot_predictions - num_corrected_predictions ) /tot_predictions

    return (accuracy, error)








