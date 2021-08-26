import numpy
import model_evaluation

def k_cross_validation_accuracy(D, L, k, classifier):
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


