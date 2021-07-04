import stats
import dimensionality_reduction_techniques as redTec
import numpy



def mcol(v):
    return v.reshape((v.size, 1))

def load(fname):
    DList = []
    labelsList = []

    with open(fname) as f:
        for line in f:
            try:
                attrs = line.split(',')[0:4]
                attrs = mcol(numpy.array([float(i) for i in attrs]))
                label = int(line.split(',')[-1].strip())
                DList.append(attrs)
                labelsList.append(label)
            except:
                print("eccezione riga 22")
                pass

    return numpy.hstack(DList), numpy.array(labelsList, dtype=numpy.int32)



if __name__ == '__main__':
    DTR, LTR = load('Data/original_features/Train.txt')
    DTE, LTE = load('Data/original_features/Test.txt')

    # DTR: Training Data
    # DTE: Evaluation Data
    # LTR: Training Labels
    # LTE: Evaluation Labels
    
    # compute statistics to analyse the data and the givem features
    #stats.compute_stats(DTR, LTR, show_figures = True)

    #principal_components= redTec.PCA(DTR, 2)

    #stats.plot_scatter(principal_components,LTR)
    #linear_discriminants = redTec.LDA(DTR,LTR, 1)
    #redTec.plotLDA(linear_discriminants, LTR, "Applied LDA")


    

    