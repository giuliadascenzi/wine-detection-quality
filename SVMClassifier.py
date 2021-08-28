import scipy.optimize
import numpy

def mRow(v):
    return v.reshape((1,v.size)) #sizes gives the number of elements in the matrix/array
def mCol(v):
    return v.reshape((v.size, 1))

def compute_lagrangian_wrapper(H):

    def compute_lagrangian(alpha):

        elle = numpy.ones(alpha.size) # 66,
        L_hat_D=0.5*( numpy.linalg.multi_dot([alpha.T, H, alpha]) ) - numpy.dot(alpha.T , mCol(elle))# 1x1
        L_hat_D_gradient= numpy.dot(H, alpha)-elle # 66x1
        
        return L_hat_D, L_hat_D_gradient.flatten() # 66, 
   
    
    return compute_lagrangian

def polinomial_kernel_elm(xi,xj, c, d, epsilon):
    xx = numpy.dot(xi.T, xj)
    return (xx+c)**d + epsilon

def polinomial_kernel(X, c, d):
    xx = numpy.dot(X.T, X)
    return (xx+c)**d 

def radial_kernel_elm(xi,xj,lam, epsilon):
    sottr= xi-xj
    return numpy.exp(-lam * (sottr*sottr).sum())+epsilon


def radial_kernel(X, lam):
    xx = numpy.zeros([X.shape[1], X.shape[1]])
    for i in range (xx.shape[0]):
        for j in range (xx.shape[1]):
            xx[i][j] = radial_kernel_elm(X[ :,i ], X[:, j], lam, epsilon=0)
    return xx

def compute_kernel_regularized_bias(k, epsilon):
    return k+epsilon    

def SVM_computeLogLikelihoods(DTR, LTR, DTE, params):  #params=[pi_T, C]
    pi_T = params[0]
    C =params[1]

    K=1
    k_values= numpy.ones([1,DTR.shape[1]]) *K
    #Creating D_hat= [xi, k] with k=1
    D_hat = numpy.vstack((DTR, k_values))
    #Creating H_hat
    # 1) creating G_hat through numpy.dot and broadcasting
    G_hat= numpy.dot(D_hat.T, D_hat)
    
    # 2)vector of the classes labels (-1/+1)
    Z = numpy.copy(LTR)
    Z[Z == 0] = -1
    Z= mCol(Z)

    
    # 3) multiply G_hat for ZiZj operating broadcasting
    H_hat= Z * Z.T * G_hat

    # Calculate L_hat_D and its gradient DUAL SOLUTION
    compute_lagr= compute_lagrangian_wrapper(H_hat)

    # Use scipy.optimize.fmin_l_bfgs_b
    x0=numpy.zeros(LTR.size) #alpha
    
    N = LTR.size #tot number of samples
    n_T = (1*(LTR==1)).sum() #num of samples belonging to the true class
    n_F = (1*(LTR==0)).sum() #num of samples belonging to the false class
    pi_emp_T = n_T / N
    pi_emp_F = n_F / N

    C_T = C * pi_T / pi_emp_T
    C_F = C * (1-pi_T) / pi_emp_F 

    bounds_list = [(0,1)] * LTR.size

    for i in range (LTR.size):
        if (LTR[i]==1):
            bounds_list[i] = (0,C_T)
        else :
            bounds_list[i] = (0,C_F)
    
    (x,f,d)= scipy.optimize.fmin_l_bfgs_b(compute_lagr, approx_grad=False, x0=x0, iprint=0, bounds=bounds_list, factr=1.0)
    
    # From the dual solution obtain the primal one
  
    # EVALUATION!

    sommatoria = mCol(x) * mCol(Z) * D_hat.T
    w_hat_star = numpy.sum( sommatoria,  axis=0 ) 
    w_star = w_hat_star[0:-1] 
    b_star = w_hat_star[-1] 
    
    scores = numpy.dot(mCol(w_star).T, DTE) + b_star

    return scores.flatten()


def Polinomial_SVM_computeLogLikelihoods(DTR, LTR, DTE, params):  #params=[pi_T, C]
    pi_T = params[0]
    C =params[1]

    
    #Creating D_hat
    D_hat = DTR
    #Creating H_hat
    # 1) creating G_hat using the polinomial kernel
    K = 0.0
    epsilon = K**2
    

    d = 2.0
    c = 0.0
    
    G_hat= polinomial_kernel(DTR,c,d)

    if (epsilon!=0):
        G_hat = compute_kernel_regularized_bias(G_hat, epsilon)
    
    
    # 2)vector of the classes labels (-1/+1)
    Z = numpy.copy(LTR)
    Z[Z == 0] = -1
    Z= mCol(Z)

    
    # 3) multiply G_hat for ZiZj operating broadcasting
    H_hat= Z * Z.T * G_hat

    # Calculate L_hat_D and its gradient DUAL SOLUTION
    compute_lagr= compute_lagrangian_wrapper(H_hat)

    # Use scipy.optimize.fmin_l_bfgs_b
    x0=numpy.zeros(LTR.size) #alpha
    
    N = LTR.size #tot number of samples
    n_T = (1*(LTR==1)).sum() #num of samples belonging to the true class
    n_F = (1*(LTR==0)).sum() #num of samples belonging to the false class
    pi_emp_T = n_T / N
    pi_emp_F = n_F / N

    C_T = C * pi_T / pi_emp_T
    C_F = C * (1-pi_T) / pi_emp_F 

    bounds_list = [(0,1)] * LTR.size

    for i in range (LTR.size):
        if (LTR[i]==1):
            bounds_list[i] = (0,C_T)
        else :
            bounds_list[i] = (0,C_F)
    
    (x,f,dd)= scipy.optimize.fmin_l_bfgs_b(compute_lagr, approx_grad=False, x0=x0, iprint=0, bounds=bounds_list, factr=1.0)
    
    # From the dual solution obtain the primal one
  
    # EVALUATION!

    scores = numpy.zeros(DTE.shape[1])
    for t in range (DTE.shape[1]): #for every sample in evaluation data (xt)
        score=0
        for i in range (DTR.shape[1]): #for every sample in test data
                score+= x[i]*Z[i]*polinomial_kernel_elm(DTR[:,i],DTE[:,t],c,d, epsilon)
        scores[t]=score

    return scores.flatten()


def RBF_SVM_computeLogLikelihoods(DTR, LTR, DTE, params):  #params=[pi_T, C, lam]
    pi_T = params[0]
    C =params[1]
    lam = params[2]

    
    #Creating D_hat
    D_hat = DTR
    #Creating H_hat
    # 1) creating G_hat using the polinomial kernel
    K = 0.0
    epsilon = K**2
    

    
    G_hat= radial_kernel(DTR,lam)

    if (epsilon!=0):
        G_hat = compute_kernel_regularized_bias(G_hat, epsilon)
    
    
    # 2)vector of the classes labels (-1/+1)
    Z = numpy.copy(LTR)
    Z[Z == 0] = -1
    Z= mCol(Z)

    
    # 3) multiply G_hat for ZiZj operating broadcasting
    H_hat= Z * Z.T * G_hat

    # Calculate L_hat_D and its gradient DUAL SOLUTION
    compute_lagr= compute_lagrangian_wrapper(H_hat)

    # Use scipy.optimize.fmin_l_bfgs_b
    x0=numpy.zeros(LTR.size) #alpha
    
    N = LTR.size #tot number of samples
    n_T = (1*(LTR==1)).sum() #num of samples belonging to the true class
    n_F = (1*(LTR==0)).sum() #num of samples belonging to the false class
    pi_emp_T = n_T / N
    pi_emp_F = n_F / N

    C_T = C * pi_T / pi_emp_T
    C_F = C * (1-pi_T) / pi_emp_F 

    bounds_list = [(0,1)] * LTR.size

    for i in range (LTR.size):
        if (LTR[i]==1):
            bounds_list[i] = (0,C_T)
        else :
            bounds_list[i] = (0,C_F)
    
    (x,f,dd)= scipy.optimize.fmin_l_bfgs_b(compute_lagr, approx_grad=False, x0=x0, iprint=0, bounds=bounds_list, factr=1.0)
    
    # From the dual solution obtain the primal one

    # EVALUATION!

    scores = numpy.zeros(DTE.shape[1])
    for t in range (DTE.shape[1]): #for every sample in evaluation data (xt)
        score=0
        for i in range (DTR.shape[1]): #for every sample in test data
                score+= x[i]*Z[i]*radial_kernel_elm(DTR[:,i],DTE[:,t],lam, epsilon)
        scores[t]=score

    return scores.flatten()