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
    D, L = load('Data/original_features/Train.txt')
    #stats.compute_stats(D, L, show_figures = False)

    principal_components= redTec.PCA(D, 2)
    print(principal_components.shape)
    #stats.plot_scatter(principal_components,L)
    linear_discriminants = redTec.LDA(D,L, 2)
    stats.plot_scatter(linear_discriminants,L)

    