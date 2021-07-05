import numpy

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