import numpy
import probability as prob

def MVG_classifier(DTR, LTR, DTE, LTE):
    log =True
    
    # 1) Compute the ML estimetes for the classifier parameters (mu, co : for each class) mu=mean, co=covariance matrix
    m = prob.means(DTR, LTR)
    c = prob.covariances(DTR, LTR, m)

    if (log==False):
        # 2) Compute the likelihood of each sample for each class and store it in a matrix #Row=classes x #columns=data_sample
        scores = prob.compute_likelihoods (DTE, m, c, 2)
        # 3) Compute class posterior probability= multiply the score matrix with prior probability
        P_c= 1/2
        classes_prior_probabilties = numpy.array([P_c, P_c, P_c]) #assuming for each class P(c)=1/3 
        S_joint = prob.compute_joint_distribution(scores, classes_prior_probabilties) 
        # 4) compute class posterior probability
        S_post = prob.compute_class_posterior_probabilities(S_joint) 
        

    if (log== True): #work with loglikelihoods
        # 2) Compute the loglikelihood of each sample for each class and store it in a matrix #Row=classes x #columns=data_sample
        scores = prob.compute_loglikelihoods (DTE, m, c, 2)
        # 3) Compute class posterior probability= multiply the score matrix with prior probability
        P_c= 1/2
        classes_prior_probabilties = numpy.array([P_c, P_c, P_c]) #assuming for each class P(c)=1/3 
        S_joint = prob.compute_joint_log_distribution(scores, classes_prior_probabilties) 
        # 4) compute class posterior probability
        S_log_post = prob.compute_log_class_posterior_probabilities(S_joint) 
        S_post = numpy.exp(S_log_post)

    # 5) predicted labels= class that has the maximum posterior probabiliti
    predicted_labels = S_post.argmax(0) #argmax over the rows = for each column selects the index of the row with the highes value
    # 6) compute accuracy and error rate
    accuracy, error = prob.compute_accuracy_error(predicted_labels, LTE) #error*100 =4.0%

    return (accuracy*predicted_labels.shape[0], predicted_labels.shape[0], predicted_labels) #return tot_true_predictions, tot_predictions




def TiedCovariance_classifier(DTR, LTR, DTE, LTE):
    log =True
     # 1) Compute the ML estimetes for the classifier parameters (mu, co : for each class) mu=mean, co=covariance matrix
    m = prob.means(DTR, LTR)
    c_MVG = prob.covariances(DTR, LTR, m)


    # 2) get the tied covariance
    csigma = numpy.zeros((10,10))

    for i in range (len(c_MVG)):
        Nc = (LTR == i).sum()
        csigma += c_MVG[i]*Nc
    csigma = csigma / LTR.shape[0]

    c =[]
    for i in range (len(c_MVG)):
        c.append(csigma)

    if (log==False):
        # 2) Compute the likelihood of each sample for each class and store it in a matrix #Row=classes x #columns=data_sample
        scores = prob.compute_likelihoods (DTE, m, c, 2)
        # 3) Compute class posterior probability= multiply the score matrix with prior probability
        P_c= 1/2
        classes_prior_probabilties = numpy.array([P_c, P_c, P_c]) #assuming for each class P(c)=1/3 
        S_joint = prob.compute_joint_distribution(scores, classes_prior_probabilties) 
        # 4) compute class posterior probability
        S_post = prob.compute_class_posterior_probabilities(S_joint) 
        

    if (log== True): #work with loglikelihoods
        # 2) Compute the loglikelihood of each sample for each class and store it in a matrix #Row=classes x #columns=data_sample
        scores = prob.compute_loglikelihoods (DTE, m, c, 2)
        # 3) Compute class posterior probability= multiply the score matrix with prior probability
        P_c= 1/2
        classes_prior_probabilties = numpy.array([P_c, P_c, P_c]) #assuming for each class P(c)=1/3 
        S_joint = prob.compute_joint_log_distribution(scores, classes_prior_probabilties) 
        # 4) compute class posterior probability
        S_log_post = prob.compute_log_class_posterior_probabilities(S_joint) 
        S_post = numpy.exp(S_log_post)

    # 5) predicted labels= class that has the maximum posterior probabiliti
    predicted_labels = S_post.argmax(0) #argmax over the rows = for each column selects the index of the row with the highes value
    # 6) compute accuracy and error rate
    accuracy, error = prob.compute_accuracy_error(predicted_labels, LTE) #error*100 =2.0%

    

    return (accuracy*predicted_labels.shape[0], predicted_labels.shape[0], predicted_labels) #return tot_true_predictions, tot_predictions



def NaiveBayes_classifier(DTR, LTR, DTE, LTE):
    log =True
     # 1) Compute the ML estimetes for the classifier parameters (mu, co : for each class) mu=mean, co=covariance matrix
    m = prob.means(DTR, LTR)
    c_MVG = prob.covariances(DTR, LTR, m)


    # 2) get the diagonal only from the matrix
    c = c_MVG
    for i in range (len(c_MVG)):
        c[i]= c_MVG[i]*numpy.eye(c_MVG[i].shape[0],c_MVG[i].shape[1])

    if (log==False):
        # 2) Compute the likelihood of each sample for each class and store it in a matrix #Row=classes x #columns=data_sample
        scores = prob.compute_likelihoods (DTE, m, c, 2)
        # 3) Compute class posterior probability= multiply the score matrix with prior probability
        P_c= 1/2
        classes_prior_probabilties = numpy.array([P_c, P_c, P_c]) #assuming for each class P(c)=1/3 
        S_joint = prob.compute_joint_distribution(scores, classes_prior_probabilties) 
        # 4) compute class posterior probability
        S_post = prob.compute_class_posterior_probabilities(S_joint) 
        

    if (log== True): #work with loglikelihoods
        # 2) Compute the loglikelihood of each sample for each class and store it in a matrix #Row=classes x #columns=data_sample
        scores = prob.compute_loglikelihoods (DTE, m, c, 2)
        # 3) Compute class posterior probability= multiply the score matrix with prior probability
        P_c= 1/2
        classes_prior_probabilties = numpy.array([P_c, P_c, P_c]) #assuming for each class P(c)=1/3 
        S_joint = prob.compute_joint_log_distribution(scores, classes_prior_probabilties) 
        # 4) compute class posterior probability
        S_log_post = prob.compute_log_class_posterior_probabilities(S_joint) 
        S_post = numpy.exp(S_log_post)

    # 5) predicted labels= class that has the maximum posterior probabiliti
    predicted_labels = S_post.argmax(0) #argmax over the rows = for each column selects the index of the row with the highes value
    # 6) compute accuracy and error rate
    accuracy, error = prob.compute_accuracy_error(predicted_labels, LTE) #error*100 =2.0%

    

    return (accuracy*predicted_labels.shape[0], predicted_labels.shape[0], predicted_labels) #return tot_true_predictions, tot_predictions
