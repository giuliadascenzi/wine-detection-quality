import numpy
import matplotlib
import matplotlib.pyplot as plt

def mcol(v):
    return v.reshape((v.size, 1))

def plot_hist(D, L, path):

    D0 = D[:, L==0]
    D1 = D[:, L==1]

    hFea = {
        0: 'fixed acidity',
        1: 'volatile acidity',
        2: 'citric acid',
        3: 'residual sugar',
        4: 'chlorides',
        5: 'free sulfur dioxide',
        6: 'total sulfur dioxide',
        7: 'density',
        8: 'pH',
        9: 'sulphates',
        10: 'alcohol'
        }

    for dIdx in range(D.shape[0]):
        plt.figure()
        plt.xlabel(hFea[dIdx])
        plt.hist(D0[dIdx, :], bins = 30, density = True, alpha = 0.4, label = 'low quality')
        plt.hist(D1[dIdx, :], bins = 30, density = True, alpha = 0.4, label = 'high quality')
        
        plt.legend()
        plt.savefig(str(path)+ '/hist_%d.png' % dIdx)
        


def plot_scatter(D, L, title):

    D0 = D[:, L==0]
    D1 = D[:, L==1]

    hFea = {
        0: 'fixed acidity',
        1: 'volatile acidity',
        2: 'citric acid',
        3: 'residual sugar',
        4: 'chlorides',
        5: 'free sulfur dioxide',
        6: 'total sulfur dioxide',
        7: 'density',
        8: 'pH',
        9: 'sulphates',
        10: 'alcohol'
        }


    for dIdx1 in range(D.shape[0]):
        for dIdx2 in range(dIdx1, D.shape[0]):
            if dIdx1 == dIdx2:
                continue
            plt.figure()
            plt.title(title+" features")
            plt.xlabel(hFea[dIdx1])
            plt.ylabel(hFea[dIdx2])
            plt.scatter(D0[dIdx1, :], D0[dIdx2, :], label = 'low quality')
            plt.scatter(D1[dIdx1, :], D1[dIdx2, :], label = 'high quality')

        
            plt.legend()
            plt.savefig('Stat/Scatter/'+title+'/scatter_%d_%d.png' % (dIdx1, dIdx2))

        
def plot_heatmaps (D, L, path):
    # show the correlation of the features in the whole dataset
    C =numpy.corrcoef(D)
    plt.figure()
    #plt.imshow(C, cmap='Blues')
    plt.imshow(C, cmap='Greens')
    plt.colorbar()
    plt.title("Whole dataset")
    plt.savefig(str(path)+'/whole_dataset')

    # show the correlation of the features in the samples of low quality wine
    C =numpy.corrcoef(D[:,L==0])
    plt.figure()
    plt.imshow(C, cmap='Greens')
    #plt.imshow(C, cmap='Oranges')
    plt.colorbar()
    plt.title("Samples of low quality wine")
    plt.savefig(str(path)+'/low_quality')

    # show the correlation of the features in the samples of high quality wine
    C =numpy.corrcoef(D[:,L==1])
    plt.figure()
    plt.imshow(C, cmap='Greens')
    plt.colorbar()
    plt.title("Samples of high quality wine")
    plt.savefig(str(path)+'/high_quality')
    

def bars_numsamples(n_high_qty, n_low_qty, title):
    
    plt.figure()
    plt.ylabel("nsamples")
    widthbar = 0.1
    plt.xticks([0,1], ['low quality', 'high quality'])
    plt.bar( 0, n_low_qty)
    plt.bar( 1,n_high_qty)
    plt.title( title + " data")
    plt.savefig('Stat/hist_number_of_data_'+title+'.png')
    

    

