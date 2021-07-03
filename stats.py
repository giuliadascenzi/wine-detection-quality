import numpy
import matplotlib
import matplotlib.pyplot as plt

def mcol(v):
    return v.reshape((v.size, 1))

def plot_hist(D, L):

    D0 = D[:, L==0]
    D1 = D[:, L==1]

    hFea = {
        0: 'Variance',
        1: 'Skewness',
        2: 'Curtosis',
        3: 'Entropy'
        }

    for dIdx in range(D.shape[0]):
        plt.figure()
        plt.xlabel(hFea[dIdx])
        plt.hist(D0[dIdx, :], bins = 10, density = True, alpha = 0.4, label = 'not authentic')
        plt.hist(D1[dIdx, :], bins = 10, density = True, alpha = 0.4, label = 'authentic')
        
        plt.legend()
        plt.tight_layout() # Use with non-default font size to keep axis label inside the figure
        plt.savefig('Stat/Hist/hist_%d.pdf' % dIdx)
        


def plot_scatter(D, L):

    D0 = D[:, L==0]
    D1 = D[:, L==1]

    hFea = {
        0: 'Variance',
        1: 'Skewness',
        2: 'Curtosis',
        3: 'Entropy'
        }


    for dIdx1 in range(D.shape[0]):
        for dIdx2 in range(D.shape[0]):
            if dIdx1 == dIdx2:
                continue
            plt.figure()
            plt.xlabel(hFea[dIdx1])
            plt.ylabel(hFea[dIdx2])
            plt.scatter(D0[dIdx1, :], D0[dIdx2, :], label = 'not authentic')
            plt.scatter(D1[dIdx1, :], D1[dIdx2, :], label = 'authentic')

        
            plt.legend()
            plt.tight_layout() # Use with non-default font size to keep axis label inside the figure
            plt.savefig('Stat/Scatter/scatter_%d_%d.pdf' % (dIdx1, dIdx2))
    
    plt.show()
        
def plot_heatmaps (D, L):
    # show the correlation of the features in the whole dataset
    C =numpy.corrcoef(D)
    plt.figure()
    plt.imshow(C, cmap='Blues')
    plt.colorbar()
    plt.title("Whole dataset")
    plt.savefig('Stat/HeatMaps/whole_dataset')

    # show the correlation of the features in the samples of authentic banknote
    C =numpy.corrcoef(D[:,L==0])
    plt.figure()
    plt.imshow(C, cmap='Greens')
    plt.colorbar()
    plt.title("Samples of authentic banknote")
    plt.savefig('Stat/HeatMaps/authentic_dataset')

    # show the correlation of the features in the samples of forged banknote
    C =numpy.corrcoef(D[:,L==1])
    plt.figure()
    plt.imshow(C, cmap='Oranges')
    plt.colorbar()
    plt.title("Samples of forged banknote")
    plt.savefig('Stat/HeatMaps/forged_dataset')
    

    
    
def compute_stats (D, L, show_figures=True):

    plot_hist(D, L)
    plot_scatter(D, L)
    

    #calculate the matrix of the Pearson product-moment correlation coefficients.
    plot_heatmaps(D, L)

    mu= D.mean(1)
    center_data= D-mcol(mu)
    plot_hist(center_data, L)

    if (show_figures):
        plt.show()
