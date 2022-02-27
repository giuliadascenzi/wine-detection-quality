import numpy
import scipy
import scipy.linalg as linalg

def mcol(v):
    return v.reshape((v.size, 1))
def mrow(v):
    return v.reshape((1, v.size))

def PCA (D, m):
    #dataset mean
    mu = D.mean(1)
    #normalize data
    normalized_data = D - mcol(mu)
    # N = number of samples
    N = D.shape[1]
    #compute the covariance matrix
    covariance_matrix= numpy.dot(normalized_data, normalized_data.T) / N

    #eigenvectors and eigenvalues
    eigenvalues,eigenvectors = numpy.linalg.eigh(covariance_matrix)
    #retrieve the first m eigenvectors
    eigenvectors_selected = eigenvectors[:, ::-1][:,0:m]

    principal_components = numpy.dot(eigenvectors_selected.T, normalized_data)
    return principal_components

def PCA_evaluation (D, m, normalized_eval):
    #dataset mean
    mu = D.mean(1)
    #normalize data
    normalized_data = D - mcol(mu)
    # N = number of samples
    N = D.shape[1]
    #compute the covariance matrix
    covariance_matrix= numpy.dot(normalized_data, normalized_data.T) / N

    #eigenvectors and eigenvalues
    eigenvalues,eigenvectors = numpy.linalg.eigh(covariance_matrix)
    #retrieve the first m eigenvectors
    eigenvectors_selected = eigenvectors[:, ::-1][:,0:m]

    principal_components = numpy.dot(eigenvectors_selected.T, normalized_data)
    principal_components_eval = numpy.dot(eigenvectors_selected.T, normalized_eval)
    return principal_components, principal_components_eval

def LDA(D,L, m):
    n_features= D.shape[0]
    class_labels = numpy.unique(L)

    within_c_cov_matrix = numpy.zeros((n_features, n_features))
    between_c_cov_matrix = numpy.zeros((n_features, n_features))

    mean_overall =D.mean(1)

    for c in class_labels:
        #calculate within class covariance matrix
        D_class = D[:, L==c]
        mean_class = D_class.mean(1)
        normalized_data_class = D_class - mcol(mean_class)
        n_samples_class = D_class.shape[1]
        cov_class= (numpy.dot(normalized_data_class, normalized_data_class.T)) /n_samples_class

        within_c_cov_matrix += cov_class * n_samples_class

        mean_diff =mcol( mean_class - mean_overall)
        between_c_cov_matrix += n_samples_class * numpy.dot(mean_diff, mean_diff.T)

    N = D.shape[1]
    within_c_cov_matrix = within_c_cov_matrix /N
    between_c_cov_matrix= between_c_cov_matrix/N

    eigenvalues, eigenvectors = linalg.eigh(between_c_cov_matrix, within_c_cov_matrix) 
    eigenvectors_chosen = eigenvectors[:, ::-1][:, 0:m]
    linear_discriminants = numpy.dot (eigenvectors_chosen.T , D)
    return linear_discriminants   

def plotLDA (data, labels, title):
    import matplotlib
    import matplotlib.pyplot as plt

    D0 = data[:, labels==0]
    D1 = data[:, labels==1]


    plt.figure()
    plt.scatter(D0, D0*0, label = 'not authentic')
    plt.scatter(D1, D1*0, label = 'authentic')

    plt.legend()
    plt.title(title)
    plt.show()










