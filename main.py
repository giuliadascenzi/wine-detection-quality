import stats
import dimensionality_reduction_techniques as redTec
import numpy
import scipy.stats

import probability as prob
import classifiers


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




def k_cross_validation(D, L, k, classifier):
    step = int(D.shape[1]/k)
    numpy.random.seed(seed=0)

    random_indexes = numpy.random.permutation(D.shape[1])

    num_correct_pred = 0

    tot = 0

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
        
        
        n,_= classifier(DTR, LTR, DEV, LEV)    

        num_correct_pred += n
        tot += LEV.size
    
    err = 1-num_correct_pred/tot

    return err










if __name__ == '__main__':
    DTR, LTR = load('Data/wine/Train.txt')
    DTE, LTE = load('Data/wine/Test.txt')

    # DTR: Training Data
    # DTE: Evaluation Data
    # LTR: Training Labels
    # LTE: Evaluation Labels
    
    # compute statistics to analyse the data and the given features
    stats.compute_stats(DTR, LTR, show_figures = True)

    #gaussianize the features
    gaussianizedFeatures = gaussianization(DTR)
    
    #stats.plot_hist(gaussianizedFeatures, LTR)


    #enstablish if data are balanced
    n_high_qty = numpy.count_nonzero(LTR == 1)
    n_low_qty = numpy.count_nonzero(LTR == 0)
    #-----> number of low qty >> number of high qty

    #PT = Prior probability for True class -> high quality
    #PF = Prior probability for False class -> low quality
    PT = 0.5
    PF = 0.5
    classes_prior_probabilties = numpy.array([PT, PF])

    _,_,predicted_labelsMVG= classifiers.MVG_classifier(DTR, LTR, DTE, LTE, classes_prior_probabilties)
    _,_,predicted_labelsTied= classifiers.TiedCovariance_classifier(DTR, LTR, DTE, LTE, classes_prior_probabilties)
    _,_,predicted_labelsNaive= classifiers.NaiveBayes_classifier(DTR, LTR, DTE, LTE, classes_prior_probabilties)

    accMVG,_ = prob.compute_accuracy_error(predicted_labelsMVG, LTE)
    accTied,_ = prob.compute_accuracy_error(predicted_labelsTied, LTE)
    accNaive,_ = prob.compute_accuracy_error(predicted_labelsNaive, LTE)

    print(accMVG, accTied, accNaive)

    #Build confusion matrix for MVG classifier
    c_matrix_MVG = prob.compute_confusion_matrix(predicted_labelsMVG, LTE, 2 )
    c_matrix_Tied = prob.compute_confusion_matrix(predicted_labelsTied, LTE, 2 )
    c_matrix_Naive = prob.compute_confusion_matrix(predicted_labelsNaive, LTE, 2 )

    
    print("Confusion matrix with MVG classifier")
    print(c_matrix_MVG)

    print("Confusion matrix with Tied classifier")
    print(c_matrix_Tied)

    print("Confusion matrix with Naive classifier")
    print(c_matrix_Naive)



    #patti llr
    m = prob.means(DTR, LTR)
    c = prob.covariances(DTR, LTR, m)
    scores = prob.compute_loglikelihoods (DTR, m, c, 2)
    llr = scores[1,:] - scores[0,:]
    #end patti llr


     #+++++++++++++++++++++++++++++++ 2) OPTIMAL BAYES DECISION ++++++++++++++++++++++++++++++++
    print(">>>OPTIMAL BAYES DECISION")
    print("PARAMETERS:")
    
    prior=0.5
    cost_fn=1
    cost_fp=1
    print("Prior class probability: "+ str(prior))
    print("Cost false negative: "+ str(cost_fn))
    print("Cost false negative: "+ str(cost_fp))
    predicted_labels,c_matrix_IP,bayes_risk,normalized_bayes_risk = prob.compute_normalised_bayes_risk_wrapper(llr, LTR, prior, cost_fn, cost_fp)
    
    print("Confusion matrix:")
    print(c_matrix_IP)
    print("Bayes risk: "+ str(bayes_risk))
    print("Normalized bayes risk: "+ str(normalized_bayes_risk))



    #principal_components= redTec.PCA(DTR, 9)
    #print(principal_components)

    #stats.plot_scatter(principal_components,LTR)
    #linear_discriminants = redTec.LDA(DTR,LTR, 1)
    #redTec.plotLDA(linear_discriminants, LTR, "Applied LDA")


   

           

