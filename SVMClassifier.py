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