import gurobipy as gp
from gurobipy import GRB
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
import math
import time
from sklearn.linear_model import LassoCV
from sklearn.linear_model import LinearRegression as LR
from sklearn.preprocessing import StandardScaler
from scipy import stats
from joblib import Parallel, delayed
from sklearn.metrics import f1_score#, matthews_corrcoef
import cvxpy as cp
import mpmath
from matplotlib.ticker import MultipleLocator
from itertools import combinations

mpmath.mp.dps = 100

# The following MCMC code is taken from https://github.com/roysaptaumich/DP-BSS/blob/main/DP-BSS-parallel.ipynb
##################################################################################
# MCMC CODE STARTS

'''
S <- stored as p-dim np array of 0,1 indicating which variables active

'''

def compare_to_rows(SS, S):
    # S <- p-dim array, SS <- (m,p) array
    # returns -1 if not present otherwise row number
    logic = np.any(np.all(SS == S, axis =1))
    if logic == True:
        return np.where(logic == True)[0][0]
    else:
        return -1


class ebreg:

    def __init__(self, tuning_parameters):

        #tuning parameters <- dictionary
        self.s = tuning_parameters['s']
        self.B = tuning_parameters['B']
        self.epsilon = tuning_parameters['epsilon']
        self.sensitivity_scaling = tuning_parameters['sensitivity_scaling']

        # MCMC parameters
        self.max_iter = tuning_parameters['max_iter']
        
        # some options
        self.standardize = tuning_parameters["standardization"]
        self.initialization = tuning_parameters["initialization"]

    def fit(self, X, y):
        self.n, self.p = X.shape
        #if self.n >= 100: self.burn_in = 500
        # standardize X
        if self.standardize is True:
            scaler1 = StandardScaler(with_mean=True, with_std=False).fit(X)
            X = scaler1.transform(X)

            y = y - y.mean()

        self.X = X
        self.y = y
        

        self.MCMC()

   

    def draw_q(self, S):
        S1 = S.copy()
        s = S.sum()
        idx_0 = np.where(S1 == 0)[0]
        idx_1 = np.where(S1 == 1)[0]
        S1[np.random.choice(idx_1, 1)] = 0
        S1[np.random.choice(idx_0, 1)] = 1

        return S1

    def OLS_pred_and_pi_n(self, S):
        X_S = self.X[:, S == 1]
        y = self.y

        reg = LR(fit_intercept=False).fit(X_S, y)  # what happens when singular
        Y_S = reg.predict(X_S)

        epsilon = self.epsilon
        Du = self.sensitivity_scaling #2 * self.ymax**2 + 2 * self.s * (self.xmax**2) *  self.sensitivity_scaling * sigma2
        # s = S.sum()
        log_pi_n =  - epsilon * np.linalg.norm(y - Y_S)**2/(Du)  # np.log(gamma + alpha/sigma2)
        return Y_S, log_pi_n
                    
    def regOLS_pred_and_pi_n(self, S):
        X_S = self.X[:, S == 1]
        y = self.y
        
        b = cp.Variable(shape = X_S.shape[1])
        constraints = [cp.norm1(b) <= self.B]
        
        obj = cp.Minimize(cp.sum_squares(X_S@b - y))

        # Form and solve problem.
        prob = cp.Problem(obj, constraints)
        prob.solve() 
        Y_S = X_S @ b.value
        Du = self.sensitivity_scaling #2 * self.ymax**2 + 2 * self.s * (self.xmax**2) *  self.sensitivity_scaling * sigma2
        # s = S.sum()
        log_pi_n =  - self.epsilon * np.linalg.norm(y - Y_S)**2/(Du)
        
        return Y_S, log_pi_n
    
    
    def initialize(self):
        
        S = np.zeros(self.p)
        
        
        if self.initialization == "Lasso":
            reg = LassoCV(n_alphas=100, fit_intercept=False,
                          cv=5, max_iter=2000).fit(self.X, self.y)
            scores = np.abs(reg.coef_)
            s_max_scores_id = np.argsort(scores)[::-1][:self.s]
            S[s_max_scores_id] = 1
        
        elif self.initialization == "MS":
            X = self.X
            y = self.y
            c = X.T@y/self.n
            c1 = np.abs(c)
            c2 = np.argsort(c1)[::-1][:self.s]
            S[c2] = 1
        
        else: S[np.random.choice(self.p, self.s, replace= False)] = 1
        
        self.initial_state = S
        
        self.S = S
        
        
        return S

    def MCMC(self):
        max_iter = self.max_iter
        
        # initialize
        S = self.initialize()
        self.S_list = [self.S]
        Y_S, log_pi_n = self.regOLS_pred_and_pi_n(self.S)
        self.Y_S_old = Y_S
        self.log_pi_n_list = [log_pi_n]
        self.RSS = np.array([np.linalg.norm(self.y - Y_S)**2/np.linalg.norm(self.y)**2])
        #self.F1 = [0]

        

        S = self.S
        y = self.y

        iter1 = 0
        no_acceptances = 0

        while (iter1 < max_iter):

            # proposal draw
            S_new = self.draw_q(S)
            Y_S_new, log_pi_n_new = self.regOLS_pred_and_pi_n(S_new)
            

            # compute hastings ratio
            try: HR = np.exp(log_pi_n_new - log_pi_n)
            except ValueError: print('Hastings ratio uncomputable')
            R = np.min([1, HR])
            if stats.uniform.rvs() <= R:
                # accept
                self.RSS = np.vstack((self.RSS, np.linalg.norm(self.y - Y_S_new)**2/np.linalg.norm(self.y)**2))
                self.S_list.pop()
                self.S_list.append(S_new)
                self.Y_S_old = Y_S_new
                S = S_new
                log_pi_n = log_pi_n_new
                no_acceptances += 1
            else:
                self.RSS = np.vstack((self.RSS, np.linalg.norm(self.y - self.Y_S_old)**2/np.linalg.norm(self.y)**2))

        

            iter1 += 1

        
        self.acceptance = no_acceptances 

# MCMC CODE ENDS
##################################################################################

def generate_input_data(n, p, sigma, s, rho,snr):
    # Generate covariance matrix Σ
    Sigma = np.zeros((p, p))
    for i in range(p):
        for j in range(p):
            Sigma[i, j] = rho**abs(i - j)

    # Generate X matrix with rows drawn from multivariate normal
    X = np.random.multivariate_normal(mean=np.zeros(p), cov=Sigma, size=n)

    # Generate true coefficients vector beta_true
    beta_true = np.zeros(p)
    indices = np.arange(1, 2*s+1, 2)  # Indices for non-zero entries
    beta_true[indices] = 1 / np.sqrt(s)
    
    snr = np.sqrt(snr)

    # Generate target vector y
    epsilon = np.random.normal(loc=0, scale=sigma, size=n)
    
    alpha = np.linalg.norm(X @ beta_true, ord=2)/(snr*np.linalg.norm(epsilon, ord=2))

    y = X @ beta_true + alpha*epsilon

    return X, y, beta_true

def get_distinct_columns(arr):
    unique_columns = set()
    distinct_indices = []
    for col_idx in range(arr.shape[1]):
        column_indices = tuple(np.nonzero(arr[:, col_idx])[0])
        if column_indices not in unique_columns:
            unique_columns.add(column_indices)
            distinct_indices.append(col_idx)
    reduced_matrix = arr[:, distinct_indices]
    return distinct_indices,reduced_matrix

def draw_index(vec, draw):
    for i in range(len(vec)):
        if draw <= vec[i] and (i == 0 or draw > vec[i-1]):
            break
    return i

def find_matching_column(beta_true, reduced_matrix, R):
    # Get the number of columns in reduced_matrix
    num_columns = reduced_matrix.shape[1]
    
    # Iterate through each column of reduced_matrix
    for i in range(num_columns):
        # Check if the current column shares the same location of nonzeros as beta_true
        if np.array_equal(beta_true.nonzero()[0], reduced_matrix[:, i].nonzero()[0]):
            return i  # Return the index of the matching column
    
    # If no column matches, return R
    return R


def least_squares(X, y, s=5, l=1.1, Lambda=100):
    
    n, p = X.shape

    # Create a Gurobi model
    model = gp.Model()

    # Add decision variable beta
    beta = model.addMVar(p, name="beta")
    theta = model.addMVar(p, name="theta")

    # Add binary decision variable z
    z = model.addMVar(p, vtype=GRB.BINARY, name="z")

    A = X.T @ X
    Q = 0.5* (A.T + A) + 1e-9 * np.identity(p)
    c = y.T @ X

    objective = (1/(2*n)) * (beta.transpose() @ Q @ beta -2* c @ beta + y.T @ y) + Lambda/(2*n)*gp.quicksum(theta[i] for i in range(p))

    model.setObjective(objective, GRB.MINIMIZE)
  
    # Add constraints |βi| <= θi * zi
    for i in range(p):
        model.addConstr(beta[i]*beta[i] <= theta[i] * z[i])

    # Add constraint sum(z) <= s
    model.addConstr(gp.quicksum(z[i] for i in range(p)) <= s)
    model.addConstr(gp.quicksum(theta[i] for i in range(p)) <= l*l)


    # Optimize the model
    model.optimize()

    # Get the optimal solution
    beta = np.array(model.getAttr('x'))[0:p]

    return beta

def forward_CD(X, y, num_sp=5, update_iter = 1):
    y = y.reshape(-1, 1)
    XTX = X.T @ X
    XTY = X.T @ y
    totp = X.shape[1]
    num_cin = totp
    num_cout = 1
    ksize = int(totp / num_cin)
    
    W = np.zeros((totp,num_cout))
    prune_list = np.zeros(num_cin)
    
    Hess_inv = np.zeros((num_cin,ksize,ksize))
    for ih in range(num_cin):
        Hess_inv[ih,:,:] = np.linalg.inv(XTX[ih*ksize:(ih+1)*ksize,ih*ksize:(ih+1)*ksize])
    prune_list = np.abs(np.sum(np.sum(W.reshape(num_cin, ksize, num_cout),axis=2),axis=1)) <= 1e-12
    for i1 in range(num_sp):
        
        G_tmp = np.zeros_like(W)
        W_tmp = np.zeros_like(W)
        for i2 in range(num_cin):
            G_tmp[i2*ksize:(i2+1)*ksize,:] = XTY[i2*ksize:(i2+1)*ksize,:] - XTX[i2*ksize:(i2+1)*ksize,:]@W + XTX[i2*ksize:(i2+1)*ksize,i2*ksize:(i2+1)*ksize]@W[i2*ksize:(i2+1)*ksize,:]
            W_tmp[i2*ksize:(i2+1)*ksize,:] = Hess_inv[i2,:,:] @ G_tmp[i2*ksize:(i2+1)*ksize,:]
        
        
        obj_cha = W_tmp * G_tmp
        obj_cha = obj_cha.reshape(num_cin, ksize, num_cout)
        obj_sum = np.sum(np.sum(obj_cha,axis=2),axis=1)

        
        idx = np.argmin(-obj_sum + 1e20*(1-prune_list))   
        W[idx*ksize:(idx+1)*ksize,:] = np.copy(W_tmp[idx*ksize:(idx+1)*ksize,:])
        
        prune_list = np.abs(np.sum(np.sum(W.reshape(num_cin, ksize, num_cout),axis=2),axis=1)) <= 1e-12
        for i2 in range(update_iter):
            for i3 in range(num_cin):
                if prune_list[i3] == True:
                    continue
                W[i3*ksize:(i3+1)*ksize,:] = Hess_inv[i3,:,:] @ (XTY[i3*ksize:(i3+1)*ksize,:]-XTX[i3*ksize:(i3+1)*ksize,:]@W + XTX[i3*ksize:(i3+1)*ksize,i3*ksize:(i3+1)*ksize]@W[i3*ksize:(i3+1)*ksize,:])

    for i2 in range(5*update_iter):
        for i3 in range(num_cin):
            if prune_list[i3] == True:
                continue
            W[i3*ksize:(i3+1)*ksize,:] = Hess_inv[i3,:,:] @ (XTY[i3*ksize:(i3+1)*ksize,:]-XTX[i3*ksize:(i3+1)*ksize,:]@W + XTX[i3*ksize:(i3+1)*ksize,i3*ksize:(i3+1)*ksize]@W[i3*ksize:(i3+1)*ksize,:])
    
    W = W.flatten()
    return W


def lasso_gurobi(X, y, Lambda=20.0):
    
    n, p = X.shape
    
    # Create a Gurobi model
    model = gp.Model()

    # Add decision variable beta
    beta = model.addMVar(p, name="beta")
    z = model.addMVar(p, name="z")

    # Add objective function (least squares)
    

    A = X.T @ X
    Q = 0.5* (A.T + A) + 1e-9 * np.identity(p)
    c = y.T @ X

    objective = (1/(2*n)) * (beta.transpose() @ Q @ beta -2* c @ beta + y.T @ y) + Lambda/n * gp.quicksum( z[j] for j in range(p))

    model.setObjective(objective, GRB.MINIMIZE)
    for i in range(p):
            model.addConstr(beta[i] <= z[i])
            model.addConstr(-z[i] <= beta[i])
    # Optimize the model
    model.optimize()

    # Get the optimal solution
    obj_value = model.objVal
    if model.status == GRB.OPTIMAL:
        beta = np.array(model.getAttr('x'))[0:p]
        return beta
    else:
        print("Optimization failed. Status code:", model.status)
        return None

def split_matrix(X,y):
    n, p = X.shape
    sqrt_n = int(math.sqrt(n))
    num_blocks = int(math.floor(sqrt_n))
    rows_per_block = [sqrt_n] * num_blocks

    # Distribute remaining rows
    leftover_rows = n - sqrt_n * sqrt_n
    for i in range(leftover_rows):
        rows_per_block[i % num_blocks] += 1

    # Create blocks
    blocks_X = []
    blocks_y = []
    start_idx = 0
    for rows_in_block in rows_per_block:
        blocks_X.append(X[start_idx:start_idx+rows_in_block, :])
        blocks_y.append(y[start_idx:start_idx+rows_in_block])
        start_idx += rows_in_block

    return blocks_X, blocks_y


def subsample(X, y, s, eps,func, num_iterations=50, Lambda=-1):
    
    n,p = X.shape
    blocks_X, blocks_y = split_matrix(X,y)
    num_blocks = len(blocks_X)
    theta = np.zeros((num_blocks,p))
    V_matrix = np.zeros((num_blocks,p))    

    for i in range(0,num_blocks):
        if Lambda != -1:
            #we are doing lasso subsampling
            theta_fw = func(blocks_X[i], blocks_y[i], Lambda)
            theta[i,:]= theta_fw
            V_matrix[i,:] = np.where(np.abs(theta_fw) > 1e-8, 1, 0)
        else:
            theta_fw = func(blocks_X[i], blocks_y[i])
            theta[i,:]= theta_fw
            sorted_indices = np.argsort(-np.abs(theta_fw))
            # Select the top s indices
            V_matrix[i,sorted_indices[:s]] = 1

    final_theta = np.zeros((p,num_iterations))
    for i in range(num_iterations):
        V_average = np.mean(V_matrix, axis=0).T + np.random.laplace(scale=2*s/(num_blocks*eps), size=p) 
        sorted_indices = np.argsort(-np.abs(V_average))
        final_theta[sorted_indices[:s],i] = 1
    return final_theta

def private_mcmc(X, y, s, eps, mcmc_iter, b_x = 1, b_y = 1, num_chains=50, l1bound = 3.5):

    B = l1bound
    Du = (B * b_x  + b_y)**2
    n,p = X.shape

    tuning_parameters = {'epsilon': eps,
                            'sensitivity_scaling': Du,
                            's'      : s,
                            'B'      : B,   
                            'max_iter': mcmc_iter,
                        'initialization': 'Random',
                        'standardization': False}

    model = ebreg(tuning_parameters)

    def fit_MCMC(i):
        model.fit(X,y)
        #print(str(i)+'th chain fitting complete')
        support_mcmc = np.zeros((p,))
        S_hat = np.where(model.S_list[-1]>0)[0]
        support_mcmc[S_hat] = 1
        return support_mcmc

    #start = time.time()
    results = Parallel(n_jobs= num_chains)(delayed(fit_MCMC)(i) for i in range(num_chains))
    mcmc_supports = np.zeros((p,num_chains))
    for i in range(num_chains):
        mcmc_supports[:,i] = results[i]
    #end = time.time()
    return mcmc_supports



def least_squares_backsolve_pgd(X, y, r=1.1, Lambda= 10e-2, max_iter = 1000):
    # Subset matrix X using these indices
    n,p = X.shape
    # obj = 0
    obj_new = np.zeros((max_iter,))
    A = X.T @ X
    b = X.T @ y
    eig = power_method(A+Lambda*np.eye(p))
    # print(eig)
    step_size = n/eig ## need to think about step size
    beta = np.zeros((X.shape[1],)) ##project just in case initial point is infeasible
    d = y.T @ y
    for i in range(max_iter):
        # Compute the gradient
        grad = 1/n*(A @ beta  - b) + Lambda/n* beta

        # Perform the gradient update
        beta_new = beta - step_size * grad

        # Project onto the l2 ball
        beta_new = project_l2_ball(beta_new,r)
        beta = beta_new
        obj_new[i] = 1/(2*n)*(beta.T @ A @ beta -2* beta.T @ b+d) + Lambda/(2*n)*beta.T @ beta
        if i>0 and abs(obj_new[i] - obj_new[i-1])/obj_new[i] < 1e-8: #if objective values are starting to converge, end the PGD
            obj_new = obj_new[0:i]
            break
    return obj_new[-1]



def power_method(A, max_iterations=1000, tol=1e-6):
    """
    Power method to find the largest eigenvalue and corresponding eigenvector of a matrix.

    Parameters:
    A (numpy.ndarray): The square matrix.
    max_iterations (int): Maximum number of iterations.
    tol (float): Tolerance for convergence.

    Returns:
    eigenvalue (float): The largest eigenvalue.
    eigenvector (numpy.ndarray): The corresponding eigenvector.
    """

    n = A.shape[0]
    x0 = np.random.rand(n)  # Random initial guess for the eigenvector
    x0 /= np.linalg.norm(x0)  # Normalize the initial guess

    for i in range(max_iterations):
        x = np.dot(A, x0)
        eigenvalue = np.dot(x, x0)
        x /= np.linalg.norm(x)  # Normalize the eigenvector
        error = np.linalg.norm(x - x0)
        if error < tol:
            break
        x0 = x

    return eigenvalue


def project_l0_ball(x, s):
    indices = np.argsort(np.abs(x))[::-1][:s]
    proj_x = np.zeros_like(x)
    proj_x[indices] = x[indices]
    return proj_x

def project_l2_ball(x, r):
    return r/max(r,np.linalg.norm(x))*x

def iht_heuristic(s, X, A,b,d, eig,zt, Lambda, max_iter = 1000):
    n = X.shape[0]
    stepsize = n/eig
    obj = 0
    obj_new = 1000
    beta = project_l0_ball(zt,s)
    for i in range(max_iter):
        grad = 1/n*(A @ beta  - b) + Lambda/n*beta
        beta_new = beta - stepsize * grad

        # Project onto the l0 ball
        beta_new = project_l0_ball(beta_new,s)
        obj = obj_new
        obj_new = 1/(2*n)*(beta_new.T @ A @ beta_new -2* beta_new.T @ b+d) + Lambda/(2*n)*np.sum(beta_new**2)
        if abs(obj - obj_new)/obj < 1e-3: #if objective values are starting to converge, end the PGD
            beta = beta_new
            zt = np.where(beta != 0, 1,beta)
            # print(f"the current zt is {zt }")
            break
        beta = beta_new
        zt = np.where(beta != 0, 1,beta)
    # print(f"iht heuristic took {i} iterations")
    return beta, zt

def compute_c_and_gradc_pgd(X,A, b, d, z, r, beta0, Lambda, max_iter = 1000):
    # Subset matrix X using these indices
    n = X.shape[0]
    obj_new = np.zeros((max_iter,))
    eig = power_method(A+Lambda*np.diag(1 / z))
    # print(eig)
    step_size = n/eig ## need to think about step size
    beta = project_l2_ball(beta0,r) ##project just in case initial point is infeasible
    for i in range(max_iter):
        # Compute the gradient
        grad = 1/n*(A @ beta  - b) + Lambda/n* (beta/z)

        # Perform the gradient update
        beta_new = beta - step_size * grad

        # Project onto the l2 ball
        beta_new = project_l2_ball(beta_new,r)

        # obj = obj_new
        obj_new[i] = 1/(2*n)*(beta_new.T @ A @ beta_new -2* beta_new.T @ b+d) + Lambda/(2*n)*np.sum(beta_new**2 / z)
        if i>0 and abs(obj_new[i] - obj_new[i-1])/obj_new[i] < 1e-8: #if objective values are starting to converge, end the PGD
            beta = beta_new
            obj_new = obj_new[0:i]
            # print(f"i exited pgd after {i} iterations")
            break
        beta = beta_new
    # print(f"i exited pgd after {i} iterations")
    c = obj_new[-1]
    gradc = -Lambda/(2*n)*np.square(beta)/np.square(z)
    return c, gradc


def generate_combinations(arr, k):
    n = len(arr)
    num_combinations = math.comb(n, k)
    result = np.empty((num_combinations, k), dtype=arr.dtype)
    
    # Helper function to recursively generate combinations
    def generate_combinations_recursive(arr, k, start, index, current_comb):
        if index == k:
            result[generate_combinations_recursive.counter] = current_comb
            generate_combinations_recursive.counter += 1
            return
        
        for i in range(start, len(arr)):
            current_comb[index] = arr[i]
            generate_combinations_recursive(arr, k, i + 1, index + 1, current_comb)
    
    generate_combinations_recursive.counter = 0
    current_comb = np.empty(k, dtype=arr.dtype)
    generate_combinations_recursive(arr, k, 0, 0, current_comb)
    
    return result


def outer_approx_gurobi_constraint_pgd_with_cuts(y, X, s, r=1.1, Lambda=10e-2, tol=5e-3, iter_pgd=1000, max_iter = 1000, per1mist = .1):
    n,p = X.shape

    env = gp.Env(empty=True)
    env.setParam("OutputFlag",0)
    env.start()
    model = gp.Model("mip1", env=env)
    z = model.addMVar(p, vtype=GRB.BINARY, name="z")
    ita = model.addVar(lb=0.0, name="ita")

    model.setObjective(ita, GRB.MINIMIZE)

    zt= np.zeros((p,))
    zt[(p-s):p] = 1     #last s entries of s1 are 1
    # print(f"the current zt is {zt }")

    #we can add the cut below but might not do much
    A = X.T @ X
    b = X.T @ y
    d = y.T @ y

    L_matrix = A + Lambda*np.eye(p)
    eig = power_method(L_matrix)
    
    beta, zt = iht_heuristic(s, X, A,b,d, eig,zt, Lambda, max_iter)
    zt[np.argsort(np.abs(zt))[:p-s]] = 0 #fixes precision errors

    one_indices = np.nonzero(zt)[0]

    # Find indices where beta_true is 0
    zero_indices = np.where(zt == 0)[0]


    corr_indices = np.argsort(np.abs(b))[::-1]  # Sort indices in descending order of absolute values
    idx_end = int(per1mist * len(b))  # Calculate the index corresponding to the top 10%
    top_10_corr_indices = corr_indices[:idx_end]  # Get the indices of top 10% elements

    idx_zeros = zero_indices[np.isin(zero_indices, top_10_corr_indices)]
    idx_ones = one_indices[np.isin(one_indices, top_10_corr_indices)]

    for k in range(1,s+1):
        idx_ones_k = generate_combinations(idx_ones, k)
        idx_zeros_k = generate_combinations(idx_zeros, k)
        for one_idx in idx_ones_k:
                for zero_idx in idx_zeros_k:
                    # Create a copy of zt
                    mistake_vector = np.copy(zt)
                    mistake_vector[one_idx] = 0
                    mistake_vector[zero_idx] = 1
                    mistake_noise = add_noise(mistake_vector)
                    bfpgd = time.time()
                    c, gradc = compute_c_and_gradc_pgd(X,A, b,d, mistake_noise, r,mistake_vector, Lambda,iter_pgd)
                    # print(f"pgd took {time.time()-bfpgd}")
                    bfpgd = time.time()
                    gradcT = np.transpose(gradc)
                    model.addConstr(c + gradcT @ (z - mistake_noise) <= ita)

                # print(f"adding cut took {time.time()-bfpgd}")
        if k== 1:
            break

    zt_hat = add_noise(zt)
    # print(f"the zt after heuristic is {zt}")
    c,gradc = compute_c_and_gradc_pgd(X,A, b,d, zt_hat, r,beta, Lambda, iter_pgd)

    gradcT = np.transpose(gradc)
    model.addConstr(c + gradcT @ (z - zt_hat) <= ita)

    # print(f"the current zt is {zt }, the #iter is {i}")
    ita_t = 0
    t = 0
    # print(f"the current cost is {c } and the current gradient is {gradc}")
    one_indices = np.nonzero(zt)[0]
    Xs = X[:, one_indices]
    curr_obj = least_squares_backsolve_pgd(Xs, y, r, Lambda)

    model.addConstr(gp.quicksum(z[i] for i in range(p)) <= s)
    while np.abs(ita_t -curr_obj)/curr_obj > 1e-3:
    # while np.abs(ita_t -c)/c > 10e-3:
        bfoa = time.time()
        model.optimize()
        # print(f"OA itself took {time.time()-bfoa}")
        ita_t = model.objVal
        t +=1
        print(f"curiter is {t}")
        if t > 1000:
            print("i got here, takes too long")
            return -1,-1,-1, -1, -1, -1
        zt= np.array(model.getAttr('x'))[0:p]
        zt[np.argsort(np.abs(zt))[:p-s]] = 0 #fixes precision errors
        # print(f"current ita is {ita_t}")
        # print(f"the current zt is {zt }")
        zt_hat = add_noise(zt)
        # print(f"the current zt_hat is {zt_hat }")
        c, gradc = compute_c_and_gradc_pgd(X,A, b,d, zt_hat, r,zt, Lambda, iter_pgd)
        gradcT = np.transpose(gradc)
        model.addConstr(c + gradcT @ (z - zt_hat) <= ita)

        one_indices = np.nonzero(zt)[0]
        Xs = X[:, one_indices]
        curr_obj = least_squares_backsolve_pgd(Xs, y, r, Lambda)
        # print(f"the current cost is {c } and the current gradient is {gradc}")
    print(f"best: {zt}")
    print(f"iterations first set: {t}")
    zt[np.argsort(np.abs(zt))[:p-s]] = 0 #fixes precision errors


    one_indices_opt = np.nonzero(zt)[0]

    # Find indices where beta_true is 0
    zero_indices = np.where(zt == 0)[0]

    # Create an empty list to store 1-mistake vectors
    betas = np.zeros((p,(p-s)*s+2))
    betas_mistakes = np.zeros((p,s+1))
    betas[:,0] = zt
    betas_mistakes[:,0] = zt

    gaps_mistakes = np.zeros((s+1,))
    gaps_mistakes[0]  = model.getAttr(GRB.Attr.MIPGap)

    obj_values_mistakes = np.zeros((s+1,))
    obj_values = np.zeros(((p-s)*s+2,))
    Xs = X[:, one_indices]
    obj_values[0] = least_squares_backsolve_pgd(Xs, y, r, Lambda)
    obj_values_mistakes[0] = obj_values[0]

    k=1
    # bfbacksolve = time.time()
    for one_idx in one_indices_opt:
        for zero_idx in zero_indices:
            # Create a copy of zt

            mistake_vector = np.copy(zt)
            # Flip the value at each index
            mistake_vector[one_idx] = 0
            mistake_vector[zero_idx] = 1
            betas[:,k] = mistake_vector
            nonzero_indices = np.nonzero(mistake_vector)[0]
            Xs = X[:, nonzero_indices]
            #backsolve happens here
            obj_values[k]= least_squares_backsolve_pgd(Xs, y, r, Lambda)
            k +=1
    # print(f"backsolving took {time.time()-bfbacksolve}")

    model.addConstr(gp.quicksum(z[i] for i in one_indices_opt) <= s-1.5)

    ita_t = 0
    t=0
    # while np.abs(ita_t -c)/c > tol:
    while np.abs(ita_t -curr_obj)/curr_obj > tol:
        bfoa = time.time()
        model.optimize()
        # print(f"OA (second set iterations) itself took {time.time()-bfoa}")
        ita_t = model.objVal
        t +=1
        print(f"current iter: {t} ")
        if t > 500:
            print("i got here during best 2 mistake, takes too long")
            return -1,-1,-1, -1, -1, -1
        zt= np.array(model.getAttr('x'))[0:p]
        zt[np.argsort(np.abs(zt))[:p-s]] = 0 #fixes precision errors
        zt_hat = add_noise(zt)
        c, gradc = compute_c_and_gradc_pgd(X,A, b,d,  zt_hat, r,zt, Lambda, iter_pgd)
        gradcT = np.transpose(gradc)
        model.addConstr(c + gradcT @ (z - zt_hat) <= ita)

        one_indices = np.nonzero(zt)[0]
        Xs = X[:, one_indices]
        curr_obj = least_squares_backsolve_pgd(Xs, y, r, Lambda)
    print(f"best with 2 mistakes: {zt}")
    print(f"iterations second set: {t}")
    zt[np.argsort(np.abs(zt))[:p-s]] = 0 #fixes precision errors
    betas[:,1+(p-s)*s] = zt
    nonzero_indices = np.nonzero(zt)[0]
    Xs = X[:, nonzero_indices]
    obj_values[1+(p-s)*s] = least_squares_backsolve_pgd(Xs, y, r, Lambda)
    

    gap = model.getAttr(GRB.Attr.MIPGap)

    if obj_values[1+(p-s)*s] >= max(obj_values[0:(p-s)*s]): #we have correctly identified top R = 1+ (p-s)*s
        sorted_indices = np.argsort(obj_values)
        sorted_obj_values = obj_values[sorted_indices]
        sorted_betas = betas[:,sorted_indices]
        
    else:
        print(f"the model obj was {obj_values[1+(p-s)*s]}, the max 1mist was {max(obj_values[0:(p-s)*s])}")
        return betas, -1, obj_values, -1,-1,-1

    #finding the rest of the mistakes
    betas_mistakes[:,1] = sorted_betas[:,1]
    betas_mistakes[:,2] = sorted_betas[:,1+(p-s)*s]
    obj_values_mistakes[1] = sorted_obj_values[1]
    obj_values_mistakes[2] = sorted_obj_values[1+(p-s)*s]
    for j in range(2,s):
        model.addConstr(gp.quicksum(z[i] for i in one_indices_opt) <= s-j-0.5)
        ita_t = 0
        t=0
        while np.abs(ita_t -curr_obj)/curr_obj > 2*tol:
            # bfoa = time.time()
            model.optimize()
            # print(f"OA (second set iterations) itself took {time.time()-bfoa}")
            ita_t = model.objVal
            t +=1
            print(f"current iter: {t} ")
            if t > 500:
                print(f"i got here during best {j+1} mistake, takes too long")
                return -1,-1,-1, -1, -1, -1
            zt= np.array(model.getAttr('x'))[0:p]
            zt[np.argsort(np.abs(zt))[:p-s]] = 0 #fixes precision errors
            zt_hat = add_noise(zt)
            c, gradc = compute_c_and_gradc_pgd(X,A, b,d,  zt_hat, r,zt, Lambda, iter_pgd)
            gradcT = np.transpose(gradc)
            model.addConstr(c + gradcT @ (z - zt_hat) <= ita)

            one_indices = np.nonzero(zt)[0]
            Xs = X[:, one_indices]
            curr_obj = least_squares_backsolve_pgd(Xs, y, r, Lambda)
        print(f"best with {j+1} mistakes: {zt}")
        print(f"iterations {j+1} set: {t}")
        zt[np.argsort(np.abs(zt))[:p-s]] = 0 #fixes precision errors
        betas_mistakes[:,j+1] = zt
        nonzero_indices = np.nonzero(zt)[0]
        Xs = X[:, nonzero_indices]
        obj_values_mistakes[j+1] = least_squares_backsolve_pgd(Xs, y, r, Lambda)

    return sorted_betas, gap, sorted_obj_values, gaps_mistakes, betas_mistakes, obj_values_mistakes


def add_noise(zt, lb=1e-3, ub=5e-3):
    zero_indices = np.where(zt == 0)[0]
    zt_hat = zt.copy()
    zt_hat[zero_indices] += np.random.uniform(lb, ub, size=len(zero_indices))
    return zt_hat

# =========================
# helpers (selection ⇄ obj-pert)
# =========================
def _clip_for_selection(X, y, b_x, b_y):
    # OA selection uses clipped data
    Xc = np.where(np.abs(X) > b_x, np.sign(X) * b_x, X)
    yc = np.where(np.abs(y) > b_y, np.sign(y) * b_y, y)
    return Xc, yc

def _delta_sensitivity(n, s, r, b_x, b_y):
    # same delta you already use before:  1/(2n)*(b_y^2 + 2 b_y b_x r sqrt(s) + b_x^2 r^2 s)
    return (b_y**2 + 2*b_y*b_x*r*np.sqrt(s) + (b_x**2)*(r**2)*s) / (2.0*n)

def _draw_from_prob(prob_vec, u):
    # prob_vec is a *probability* vector (not CDF)
    # returns index ~ prob_vec
    cum = np.cumsum(prob_vec)
    return int(np.searchsorted(cum, u, side='right'))

def _draw_topk_and_mistakes(reduced_matrix, betas_mistakes, prob, prob2, p, s, R_eff, num_draws):
    """Return two arrays (p × num_draws) of supports for Top-R and Mistakes."""
    reduced_matrix = np.where(np.array(reduced_matrix) != 0, 1, 0)
    betas_mistakes = np.where(np.array(betas_mistakes) != 0, 1, 0)

    topk_draws     = np.zeros((p, num_draws), dtype=int)
    mistakes_draws = np.zeros((p, num_draws), dtype=int)

    for i in range(num_draws):
        # --- Top-R draw ---
        idx_top = _draw_from_prob(prob, np.random.uniform(0, 1))
        if idx_top != R_eff:
            topk_draws[:, i] = reduced_matrix[:, idx_top]
        else:
            # draw a fresh unseen s-subset
            flag = 0
            while flag == 0:
                shuf = np.arange(p); np.random.shuffle(shuf)
                nz = shuf[:s]
                cand = np.zeros((p,), dtype=int); cand[nz] = 1
                for l in range(0, R_eff):
                    if np.array_equal(reduced_matrix[:, l], cand): break
                    if l == R_eff - 1: flag = 1
            topk_draws[:, i] = cand

        # --- Mistakes draw ---
        idx_mis = _draw_from_prob(prob2, np.random.uniform(0, 1))
        mistakes_draws[:, i] = betas_mistakes[:, idx_mis]

    return topk_draws, mistakes_draws


def _select_supports(X_sel, y_sel, X_full, y_full,
                     n, p, s, eps, Lambda_ridge, r,
                     tolr, iters_pgd, perc1mist, R, 
                     b_x, b_y, num_iterations,
                     Lambda, l1bound, mcmc_iter):
    """
    Runs OA (on clipped data) to get reduced candidates + buckets, constructs the
    Top-R and Mistakes probabilities, then **draws supports** for all four methods:
      - Top-R
      - Mistakes
      - Lasso subsampling
      - Private MCMC
    Returns:
      dict with keys:
        - oops (bool)
        - opt_support           : length-p 0/1 optimal support from OA
        - supports_topk         : p × num_iterations
        - supports_mistakes     : p × num_iterations
        - supports_lasso        : p × num_iterations
        - supports_mcmc         : p × num_iterations  (resampled if needed)
    """
    # ---- OA (uses clipped data) ----
    betas_sorted, gap, obj_values, gaps_mistakes, betas_mistakes, obj_values_mistakes = \
        outer_approx_gurobi_constraint_pgd_with_cuts(
            y_sel, X_sel, s, r, Lambda_ridge, tol=tolr, iter_pgd=iters_pgd, per1mist=perc1mist
        )
    if gap == -1:
        return dict(oops=True)

    # reduced candidates and the OA-optimal support
    reduced_matrix = betas_sorted                      # p × ((p-s)s + 2)
    opt_support    = np.where(reduced_matrix[:, 0] != 0, 1, 0)

    # ---- probabilities for Top-R and Mistakes buckets ----
    R_eff = (p - s) * s + 2 if R is None else R
    delta = _delta_sensitivity(n, s, r, b_x, b_y)

    exponents = np.zeros((R_eff + 1,))
    exponents[0:R_eff] = -eps * obj_values[:R_eff] / (2 * delta)
    exponents[R_eff]   = -eps * obj_values[R_eff-1] / (2 * delta)

    exp_terms = np.array([mpmath.exp(x) for x in exponents], dtype=object)
    exp_terms[R_eff] = (math.comb(p, s) - R_eff) * exp_terms[R_eff]
    denom = mpmath.fsum(exp_terms)
    prob  = np.array(exp_terms, dtype=float) / float(denom)

    exponents2 = -eps * obj_values_mistakes / (2 * delta)
    exp_terms2 = np.array([mpmath.exp(x) for x in exponents2], dtype=object)
    for i in range(0, s+1):
        exp_terms2[i] = math.comb(p - s, i) * math.comb(s, i) * exp_terms2[i]
    denom2 = mpmath.fsum(exp_terms2)
    prob2  = np.array(exp_terms2, dtype=float) / float(denom2)

    # ---- Top-R & Mistakes: DRAW supports (shared num_iterations) ----
    supports_topk, supports_mistakes = _draw_topk_and_mistakes(
        reduced_matrix=reduced_matrix,
        betas_mistakes=betas_mistakes,
        prob=prob, prob2=prob2, p=p, s=s, R_eff=R_eff,
        num_draws=num_iterations
    )

    # ---- Lasso subsampling: already returns p × num_iterations ----
    supports_lasso = subsample(X_full, y_full, s, eps, lasso_gurobi,
                               num_iterations=num_iterations, Lambda=Lambda)

    # ---- Private MCMC: build a pool p × num_chains, then resample to num_iterations ----
    supports_mcmc   = private_mcmc(X_sel, y_sel, s, eps, mcmc_iter, b_x, b_y,
                               num_chains=num_iterations, l1bound=l1bound)

    return dict(
        oops=False,
        opt_support=opt_support,
        supports_topk=supports_topk.astype(int),
        supports_mistakes=supports_mistakes.astype(int),
        supports_lasso=supports_lasso.astype(int),
        supports_mcmc=supports_mcmc.astype(int),
    )

def _compute_utility_loss_batch(X_pre, y_pre, eps, Lambda_ridge, r, selection):
    """
    Given *preprocessed* (unclipped) data and a selection dict that already
    contains drawn supports for all four methods (each p × T), compute the Obj-Pert
    utility-loss gaps for each draw, per method. Uses selection['opt_support'] as
    the non-private baseline support for problem A.
    Returns:
      gap_topk, gap_mistakes, gap_lasso, gap_mcmc  (each length T)
    """
    opt_support      = selection['opt_support']
    S_topk           = selection['supports_topk']
    S_mistakes       = selection['supports_mistakes']
    S_lasso          = selection['supports_lasso']
    S_mcmc           = selection['supports_mcmc']

    T = S_topk.shape[1]
    gap_topk     = np.zeros((T,))
    gap_mistakes = np.zeros((T,))
    gap_lasso    = np.zeros((T,))
    gap_mcmc     = np.zeros((T,))

    for i in range(T):
        gap_topk[i]     = objpert_least_squares(X_pre, y_pre, S_topk[:, i],     eps, Lambda_ridge, r, opt_support)
        gap_mistakes[i] = objpert_least_squares(X_pre, y_pre, S_mistakes[:, i], eps, Lambda_ridge, r, opt_support)
        gap_lasso[i]    = objpert_least_squares(X_pre, y_pre, S_lasso[:, i],    eps, Lambda_ridge, r, opt_support)
        gap_mcmc[i]     = objpert_least_squares(X_pre, y_pre, S_mcmc[:, i],     eps, Lambda_ridge, r, opt_support)

    return gap_mistakes, gap_topk, gap_lasso, gap_mcmc


# A light-weight Obj-Pert that returns beta_priv (for test-MSE). Keeps your noise draw and QP, but returns coefficients.
def _objpert_beta(X, y, support_vec, eps, gamma, r_bound, rng=np.random.default_rng()):
    active_idx = np.nonzero(support_vec)[0]
    Xs = X[:, active_idx]
    n, s = Xs.shape[0], Xs.shape[1]

    lam   = s
    zeta  = 2 * (s ** 1.5)
    Delta = 2 * lam / eps

    radius    = rng.gamma(shape=s, scale=2 * zeta / eps)
    direction = rng.standard_normal(s); direction /= np.linalg.norm(direction)
    b = radius * direction

    XtX = Xs.T @ Xs
    Xty = Xs.T @ y
    I   = np.eye(s)

    c_loss  = 1.0 / (2 * n)
    c_gamma = gamma / (2 * n)
    c_Delta = Delta  / (2 * n)
    c_noise = 1.0 / n

    m = gp.Model("ObjPert-beta")
    m.Params.OutputFlag = 0

    theta = m.addMVar(s, lb=-GRB.INFINITY, name="theta")
    Q     = c_loss * XtX + (c_gamma + c_Delta) * I
    quad  = 0.5 * theta @ (2 * Q) @ theta
    lin   = (-2.0 * c_loss * Xty + c_noise * b) @ theta
    m.setObjective(quad + lin)

    # l2 ball
    m.addConstr(gp.quicksum(theta[i] * theta[i] for i in range(s)) <= r_bound * r_bound)
    m.optimize()

    theta_priv = theta.X.copy()
    beta_priv  = np.zeros((X.shape[1],))
    beta_priv[active_idx] = theta_priv
    return beta_priv

def _compute_test_mse_batch(X_train_pre, y_train_pre, X_test, y_test,
                            eps, Lambda_ridge, r, selection):
    """
    Obj-Pert on TRAIN(preprocessed), evaluate MSE on TEST(raw). Selection already
    contains drawn supports for all four methods (each p × T).
    Returns:
      mse_topk, mse_mistakes, mse_lasso, mse_mcmc  (each length T)
    """
    S_topk     = selection['supports_topk']
    S_mistakes = selection['supports_mistakes']
    S_lasso    = selection['supports_lasso']
    S_mcmc     = selection['supports_mcmc']

    T = S_topk.shape[1]
    mse_topk     = np.zeros((T,))
    mse_mistakes = np.zeros((T,))
    mse_lasso    = np.zeros((T,))
    mse_mcmc     = np.zeros((T,))

    for i in range(T):
        beta_top   = _objpert_beta(X_train_pre, y_train_pre, S_topk[:, i],     eps, Lambda_ridge, r)
        beta_mis   = _objpert_beta(X_train_pre, y_train_pre, S_mistakes[:, i], eps, Lambda_ridge, r)
        beta_lasso = _objpert_beta(X_train_pre, y_train_pre, S_lasso[:, i],    eps, Lambda_ridge, r)
        beta_mcmc  = _objpert_beta(X_train_pre, y_train_pre, S_mcmc[:, i],     eps, Lambda_ridge, r)

        mse_topk[i]     = np.mean((y_test - X_test @ beta_top)   ** 2)
        mse_mistakes[i] = np.mean((y_test - X_test @ beta_mis)   ** 2)
        mse_lasso[i]    = np.mean((y_test - X_test @ beta_lasso) ** 2)
        mse_mcmc[i]     = np.mean((y_test - X_test @ beta_mcmc)  ** 2)

    return mse_mistakes, mse_topk, mse_lasso, mse_mcmc





def objpert_least_squares(
        X, y,                     # n × p NumPy arrays (already clipped/pre‑processed)
        support_vec,              # length‑p boolean / {0,1} mask of active features
        eps,                      # ε  (ignore δ)
        gamma,                    # γ   (ridge weight in problem A)
        r_bound,                  # radius ‖θ‖₂ ≤ r_bound that defines 𝔽
        opt_support_vec,
        rng=np.random.default_rng()
    ):
    """
    Implements the modified Obj‑Pert algorithm you described,
    with   ζ = 2 s^{3/2},   λ = s,   Δ = 2 λ / ε,
    and solves both the private and non‑private QPs in Gurobi.

    Returns
    -------
    theta_priv : ndarray, shape (s,)
        Private parameter vector.
    theta_hat  : ndarray, shape (s,)
        Non‑private minimiser (problem A).
    gap        : float
        Objective value of problem A at θ_priv  minus  optimum value.
        (= prediction‑accuracy loss)
    """

    # ------------------------------------------------------------
    # 0.  Restrict the data to the active support
    # ------------------------------------------------------------
    active_idx  = np.nonzero(support_vec)[0]
    Xs_p          = X[:, active_idx]          # n × s   (s = |support|)
 
    p, n, s     = X.shape[1], X.shape[0], Xs_p.shape[1]

    # ------------------------------------------------------------
    # 1.  Privacy hyper‑parameters  λ, ζ, Δ
    # ------------------------------------------------------------
    lam    = s
    zeta   = 2 * s ** 1.5
    Delta  = 2 * lam / eps

    # ------------------------------------------------------------
    # 2.  Draw b  ~  ν₁(b; ε, ζ)  (multivariate Laplace)          
    #     R  ~  Gamma(k=s, θ = 2ζ/ε);     u  ~  uniform on S^{s‑1}
    # ------------------------------------------------------------
    radius   = rng.gamma(shape=s, scale=2 * zeta / eps)
    direction= rng.standard_normal(s);  direction /= np.linalg.norm(direction)
    b        = radius * direction

    # ------------------------------------------------------------
    # 3.  Auxiliary matrices for the quadratic forms
    # ------------------------------------------------------------
    XtX_p      = Xs_p.T @ Xs_p                         # s × s
    Xty_p      = Xs_p.T @ y                          # s‑vector
    I_s      = np.eye(s)

    #  Constant coefficients that reappear:
    c_loss   = 1.0 / (2 * n)
    c_gamma  = gamma / (2 * n)
    c_Delta  = Delta  / (2 * n)
    c_noise  = 1.0 / n

    # ------------------------------------------------------------
    # 4.  Build & solve ***private*** optimisation in Gurobi
    # ------------------------------------------------------------
    m_priv = gp.Model("ObjPert-private")
    m_priv.Params.OutputFlag = 0                 # silent

    theta = m_priv.addMVar(s, lb=-GRB.INFINITY, name="theta")

    # Objective:   ½/n ‖y - Xθ‖² + (γ+Δ)/(2n)‖θ‖² + bᵀθ / n
    Q_priv = c_loss * XtX_p + (c_gamma + c_Delta) * I_s
    quadexpr_priv = 0.5 * theta @ (2 * Q_priv) @ theta      # Gurobi wants ½ θᵀ Q θ
    linexpr_priv  = (-2.0* c_loss * Xty_p + c_noise * b) @ theta
    m_priv.setObjective(quadexpr_priv + linexpr_priv)

    # Feasible set: ‖θ‖₂ ≤ r_bound
    m_priv.addConstr(gp.quicksum(theta[i] * theta[i] for i in range(s)) <= r_bound * r_bound)

    m_priv.optimize()
    theta_priv = theta.X.copy()
    
    active_idx  = np.nonzero(opt_support_vec)[0]
    Xs_np          = X[:, active_idx]          # n × s   (s = |support|)

    # ------------------------------------------------------------
    # 5.  Build & solve ***non‑private*** problem A
    #      (no Δ term, no noise term)
    # ------------------------------------------------------------
    XtX_np      = Xs_np.T @ Xs_np                         # s × s
    Xty_np      = Xs_np.T @ y                          # s‑vector

    m_np = gp.Model("ObjPert-nonprivate")
    m_np.Params.OutputFlag = 0

    theta2 = m_np.addMVar(s, lb=-GRB.INFINITY, name="theta")

    Q_np        = c_loss * XtX_np + c_gamma * I_s
    quadexpr_np = 0.5 * theta2 @ (2 * Q_np) @ theta2
    linexpr_np  = (-2.0*c_loss * Xty_np) @ theta2

    m_np.setObjective(quadexpr_np + linexpr_np)
    m_np.addConstr(gp.quicksum(theta2[i] * theta2[i] for i in range(s)) <= r_bound * r_bound)

    m_np.optimize()
    theta_hat = theta2.X.copy()
    
    # ------------------------------------------------------------
    # 6.  Evaluate problem A's objective at both solutions
    # ------------------------------------------------------------
    
    obj_np  = c_loss * np.linalg.norm(y - Xs_np @ theta_hat)**2 \
              + c_gamma * np.linalg.norm(theta_hat)**2 
    obj_priv = c_loss * np.linalg.norm(y - Xs_p @ theta_priv)**2 \
              + c_gamma * np.linalg.norm(theta_priv)**2 
    gap = obj_priv - obj_np

    return  gap

def preprocess_objpert(X: np.ndarray, y: np.ndarray, s: float | int):
    """
    Implements steps 3–4 of the algorithm snippet:
      3.  Clip each y_i into [-s, s].
      4.  For every row X_i, look at its top-|s| coordinates (absolute value).
         If the ℓ₂‑norm of that s‑sparse vector ≥ √s, rescale the whole row so
         that ‖X_i‖₂ = √s.

    Parameters
    ----------
    X : ndarray, shape (n_samples, n_features)
    y : ndarray, shape (n_samples,)
    s : int or float
        If you follow the paper literally, s is an integer (s‑sparsity).
        It can be passed as float here – only its value is used.

    Returns
    -------
    X_new : ndarray, shape (n_samples, n_features)
    y_new : ndarray, shape (n_samples,)
    """
    # ----- step 3 : clip y ---------------------------------------------------
    y_new = np.clip(y, -s, s)

    # ----- step 4 : row‑wise sparsity check & possible rescaling ------------
    X_new = X.astype(float, copy=True)
    n, p = X_new.shape
    s_int = int(s)                       # how many coordinates to inspect
    sqrt_s = np.sqrt(s)

    for i in range(n):
        row = X_new[i]

        # indices of the |s| largest‑magnitude entries in this row
        if s_int >= p:                                   # sparse set = whole row
            top_idx = np.arange(p)
        else:
            # argpartition is O(p); ‑s gets the largest magnitudes
            top_idx = np.argpartition(np.abs(row), -s_int)[-s_int:]

        v_i = row[top_idx]
        if np.linalg.norm(v_i, 2) >= sqrt_s:
            row_norm = np.linalg.norm(row, 2)
            if row_norm > 0:                             # avoid division by zero
                X_new[i] = (sqrt_s / row_norm) * row

    return X_new, y_new
# =========================
# unified entry-point
# =========================
def run_single_instance(
    n, p, s, snr, eps,
    Lambda_ridge = 100,
    tolr=5e-3, iters_pgd=5000, perc1mist=.1, Lambda=20.0,
    seed_num = -1, R=None, rho = 0.1, sigma = 0.1,  r = 1.1,
    b_x = 1, b_y = 1,
    num_iterations = 50,
    l1bound = 3.5, mcmc_iter = 5000,
    train_test = False,     # compute test MSE (requires split, OA on *clipped* train, Obj-Pert on preprocessed train)
    util_loss  = False,     # compute utility loss (OA on *clipped* full, Obj-Pert on preprocessed full)
    frac_eps = 0.5
):
    """
    Behavior:
      - train_test=False, util_loss=False  → selection only (drawn supports for all 4 methods)
      - train_test=False, util_loss=True   → + util-loss (uses drawn supports)
      - train_test=True,  util_loss=False  → + test-MSE (uses drawn supports from TRAIN)
      - train_test=True,  util_loss=True   → both passes (full for util-loss, train/test for test-MSE)
    """
    if train_test == 0 and util_loss == 0:
        frac_eps = 1.0  # all privacy budget to selection

    if (seed_num == -1):
        np.random.seed()
    else:
        np.random.seed(seed_num)

    # ---------- data ----------
    X, y, beta_true = generate_input_data(n, p, sigma, s, rho, snr)
    results = {'beta_true': np.where(beta_true != 0, 1, 0).astype(int)}

    # ==========================================================
    # A) FULL-DATA PASS (for util_loss or bare selection)
    # ==========================================================
    need_full_selection = (not train_test) or util_loss
    if need_full_selection:
        X_sel_full, y_sel_full = _clip_for_selection(X, y, b_x, b_y)  # selection uses clipped data

        sel_full = _select_supports(
            X_sel=X_sel_full, y_sel=y_sel_full, X_full=X, y_full=y,
            n=n, p=p, s=s, eps=frac_eps*eps, Lambda_ridge=Lambda_ridge, r=r,
            tolr=tolr, iters_pgd=iters_pgd, perc1mist=perc1mist, R=R,
            b_x=b_x, b_y=b_y, num_iterations=num_iterations,
            Lambda=Lambda, l1bound=l1bound, mcmc_iter=mcmc_iter
        )
        results['selection_full'] = None if sel_full.get('oops', False) else sel_full

        if util_loss and (results['selection_full'] is not None):
            # preprocessed (unclipped) for Obj-Pert gap evaluation
            X_pre_full, y_pre_full = preprocess_objpert(X, y, s)
            gaps = _compute_utility_loss_batch(
                X_pre_full, y_pre_full, (1 - frac_eps) * eps, Lambda_ridge, r,
                selection=results['selection_full']
            )
            results['util_loss'] = dict(
                gap_mistakes=gaps[0], gap_topk=gaps[1], gap_lasso=gaps[2], gap_mcmc=gaps[3]
            )
        else:
            results['util_loss'] = None

    # ==========================================================
    # B) TRAIN/TEST PASS (for test MSE)
    # ==========================================================
    if train_test:
        from sklearn.model_selection import train_test_split
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.30, shuffle=True)

        X_sel_tr, y_sel_tr = _clip_for_selection(X_tr, y_tr, b_x, b_y)  # selection on TRAIN (clipped)
        sel_tr = _select_supports(
            X_sel=X_sel_tr, y_sel=y_sel_tr, X_full=X_tr, y_full=y_tr,
            n=X_tr.shape[0], p=p, s=s, eps=frac_eps*eps, Lambda_ridge=Lambda_ridge, r=r, 
            tolr=tolr, iters_pgd=iters_pgd, perc1mist=perc1mist, R=R,
            b_x=b_x, b_y=b_y, num_iterations=num_iterations,
            Lambda=Lambda, l1bound=l1bound, mcmc_iter=mcmc_iter
        )

        if not sel_tr.get('oops', False):
            X_pre_tr, y_pre_tr = preprocess_objpert(X_tr, y_tr, s)  # Obj-Pert on TRAIN(preprocessed), MSE on TEST(raw)
            mses = _compute_test_mse_batch(
                X_train_pre=X_pre_tr, y_train_pre=y_pre_tr, X_test=X_te, y_test=y_te,
                eps=(1 - frac_eps) * eps, Lambda_ridge=Lambda_ridge, r=r,
                selection=sel_tr
            )
            results['test_mse'] = dict(
                mse_mistakes=mses[0], mse_topk=mses[1], mse_lasso=mses[2], mse_mcmc=mses[3]
            )
            results['selection_train'] = sel_tr
        else:
            results['selection_train'] = None
            results['test_mse'] = None

    return results

def _scores_from_draws(beta_true01, S, average=True):
    """
    beta_true01: length-p 0/1
    S: p × T supports (0/1) for a method
    Returns:
      frac_correct, f1_mean  (if average=True; else per-draw arrays)
    """
    T = S.shape[1]
    truth = beta_true01.astype(int)
    eq = np.all(S.T == truth[None, :], axis=1).astype(float)  # T
    f1s = np.array([f1_score(truth, S[:, i]) for i in range(T)])
    if average:
        return eq.mean(), f1s.mean()
    return eq, f1s


def frac_f1_over_n(n_array, p, s, snr, eps,
                   Lambda_ridge=100, tolr=5e-3, iters_pgd=5000, perc1mist=.1,
                   R=None, num_trials=10, Lambda=20.0, rho=0.1, sigma=0.1, r=1.1,
                   b_x=1.0, b_y=1.0, seed_num=-1, num_iterations=100, num_chains=None,
                   l1bound=3.5, mcmc_iter=5000):
    """
    Returns dict with per-method arrays of means and standard errors over n.
    Methods: 'topk', 'mistakes', 'lasso', 'mcmc'
    """
    if num_chains is None:
        num_chains = num_iterations

    methods = ['topk', 'mistakes', 'lasso', 'mcmc']
    frac_mat = {m: np.zeros((len(n_array), num_trials)) for m in methods}
    f1_mat   = {m: np.zeros((len(n_array), num_trials)) for m in methods}

    for j, n in enumerate(n_array):
        t_done = 0
        while t_done < num_trials:
            res = run_single_instance(
                n, p, s, snr, eps,
                Lambda_ridge=Lambda_ridge, tolr=tolr, iters_pgd=iters_pgd, perc1mist=perc1mist, Lambda=Lambda,
                seed_num=seed_num, R=R, rho=rho, sigma=sigma, r=r,
                b_x=b_x, b_y=b_y,
                num_iterations=num_iterations,
                l1bound=l1bound, mcmc_iter=mcmc_iter,
                train_test=False, util_loss=False, frac_eps=1.0
            )
            sel = res.get('selection_full', None)
            if sel is None:
                continue

            beta_true01 = res['beta_true']
            # collect supports
            supports = {
                'topk':     sel['supports_topk'],
                'mistakes': sel['supports_mistakes'],
                'lasso':    sel['supports_lasso'],
                'mcmc':     sel['supports_mcmc'],
            }
            # per method scores
            for m in methods:
                frac, f1m = _scores_from_draws(beta_true01, supports[m], average=True)
                frac_mat[m][j, t_done] = frac
                f1_mat[m][j, t_done]   = f1m
            t_done += 1

    # aggregate
    out = dict()
    for m in methods:
        mean_frac = frac_mat[m].mean(axis=1)
        se_frac   = frac_mat[m].std(axis=1, ddof=1) / np.sqrt(num_trials)
        mean_f1   = f1_mat[m].mean(axis=1)
        se_f1     = f1_mat[m].std(axis=1, ddof=1) / np.sqrt(num_trials)
        out[m] = dict(mean_frac=mean_frac, se_frac=se_frac,
                      mean_f1=mean_f1, se_f1=se_f1)
    return out


def frac_f1_over_lambda(lambda_array, n, p, s, snr, eps,
                        tolr=5e-3, iters_pgd=5000, perc1mist=.1,
                        R=None, num_trials=10, Lambda=20.0, rho=0.1, sigma=0.1, r=1.1,
                        b_x=1.0, b_y=1.0, seed_num=-1, num_iterations=100, num_chains=None,
                        l1bound=3.5, mcmc_iter=5000):
    if num_chains is None:
        num_chains = num_iterations

    methods = ['topk', 'mistakes', 'lasso', 'mcmc']
    frac_mat = {m: np.zeros((len(lambda_array), num_trials)) for m in methods}
    f1_mat   = {m: np.zeros((len(lambda_array), num_trials)) for m in methods}

    for j, Lambda_ridge in enumerate(lambda_array):
        t_done = 0
        while t_done < num_trials:
            res = run_single_instance(
                n, p, s, snr, eps,
                Lambda_ridge=Lambda_ridge, tolr=tolr, iters_pgd=iters_pgd, perc1mist=perc1mist, Lambda=Lambda,
                seed_num=seed_num, R=R, rho=rho, sigma=sigma, r=r,
                b_x=b_x, b_y=b_y,
                num_iterations=num_iterations, 
                l1bound=l1bound, mcmc_iter=mcmc_iter,
                train_test=False, util_loss=False, frac_eps=1.0
            )
            sel = res.get('selection_full', None)
            if sel is None:
                continue

            beta_true01 = res['beta_true']
            supports = {
                'topk':     sel['supports_topk'],
                'mistakes': sel['supports_mistakes'],
                'lasso':    sel['supports_lasso'],
                'mcmc':     sel['supports_mcmc'],
            }
            for m in methods:
                frac, f1m = _scores_from_draws(beta_true01, supports[m], average=True)
                frac_mat[m][j, t_done] = frac
                f1_mat[m][j, t_done]   = f1m
            t_done += 1

    out = dict()
    for m in methods:
        mean_frac = frac_mat[m].mean(axis=1); se_frac = frac_mat[m].std(axis=1, ddof=1) / np.sqrt(num_trials)
        mean_f1   = f1_mat[m].mean(axis=1);   se_f1   = f1_mat[m].std(axis=1, ddof=1)   / np.sqrt(num_trials)
        out[m] = dict(mean_frac=mean_frac, se_frac=se_frac, mean_f1=mean_f1, se_f1=se_f1)
    return out

def _catalogue_supports_bruteforce(X_sel, y_sel, p, s, r, Lambda_ridge):
    idx_all = list(range(p))
    combs   = list(combinations(idx_all, s))              # C = comb(p, s)
    C       = len(combs)

    supports = np.zeros((p, C), dtype=int)
    objs     = np.zeros((C,), dtype=float)

    for k, tup in enumerate(combs):
        supports[np.array(tup), k] = 1
        Xs = X_sel[:, tup]
        objs[k] = least_squares_backsolve_pgd(Xs, y_sel, r=r, Lambda=Lambda_ridge)

    order_idx      = np.argsort(objs)
    reduced_matrix = supports[:, order_idx]
    obj_values     = objs[order_idx]
    opt_support    = reduced_matrix[:, 0].astype(int)

    # best in each "i mistakes" bucket (Hamming distance 2i)
    s_int = s
    betas_mistakes      = np.zeros((p, s_int+1), dtype=int)
    obj_values_mistakes = np.zeros((s_int+1,), dtype=float)
    hamming = np.sum(np.abs(reduced_matrix - opt_support[:, None]), axis=0)
    for i in range(0, s_int+1):
        cols = np.where(hamming == 2*i)[0]
        jmin = cols[np.argmin(obj_values[cols])]
        betas_mistakes[:, i]   = reduced_matrix[:, jmin]
        obj_values_mistakes[i] = obj_values[jmin]
    return reduced_matrix, obj_values, opt_support, betas_mistakes, obj_values_mistakes


def frac_f1_over_R(R_array, n, p, s, snr, eps,
                   Lambda_ridge=100, tolr=5e-3, iters_pgd=5000, perc1mist=.1,
                   num_trials=10, Lambda=20.0, rho=0.1, sigma=0.1, r=1.1,
                   b_x=1.0, b_y=1.0, seed_num=-1, num_iterations=100, num_chains=None,
                   l1bound=3.5, mcmc_iter=5000):
    
    methods  = ['topk', 'mistakes']
    frac_mat = {m: np.zeros((len(R_array), num_trials)) for m in methods}
    f1_mat   = {m: np.zeros((len(R_array), num_trials)) for m in methods}

    for t in range(num_trials):
        X, y, beta_true = generate_input_data(n, p, sigma, s, rho, snr)
        truth = (beta_true != 0).astype(int)

        X_sel, y_sel = _clip_for_selection(X, y, b_x, b_y)
        reduced_matrix, obj_values, opt_support, betas_mistakes, obj_values_mistakes = \
            _catalogue_supports_bruteforce(X_sel, y_sel, p, s, r, Lambda_ridge)

        delta = _delta_sensitivity(n, s, r, b_x, b_y)

        exp_terms2 = np.array([mpmath.exp(-eps * v / (2 * delta)) for v in obj_values_mistakes], dtype=object)
        for i in range(0, s+1):
            exp_terms2[i] = math.comb(p - s, i) * math.comb(s, i) * exp_terms2[i]
        denom2 = mpmath.fsum(exp_terms2)
        prob2  = np.array(exp_terms2, dtype=float) / float(denom2)

        for j, R in enumerate(R_array):
            R_eff = int(R)
            exponents = np.zeros((R_eff + 1,))
            exponents[0:R_eff] = -eps * obj_values[:R_eff] / (2 * delta)
            exponents[R_eff]   = -eps * obj_values[R_eff-1] / (2 * delta)

            exp_terms = np.array([mpmath.exp(x) for x in exponents], dtype=object)
            exp_terms[R_eff] = (math.comb(p, s) - R_eff) * exp_terms[R_eff]
            denom = mpmath.fsum(exp_terms)
            prob  = np.array(exp_terms, dtype=float) / float(denom)

            S_topk, S_mist = _draw_topk_and_mistakes(
                reduced_matrix=reduced_matrix, betas_mistakes=betas_mistakes,
                prob=prob, prob2=prob2, p=p, s=s, R_eff=R_eff, num_draws=num_iterations
            )

            bundles = {'topk': S_topk, 'mistakes': S_mist}
            for m, S in bundles.items():
                frac, f1m = _scores_from_draws(truth, S, average=True)
                frac_mat[m][j, t] = frac
                f1_mat[m][j, t]   = f1m

    out = {}
    for m in methods:
        mean_frac = frac_mat[m].mean(axis=1); se_frac = frac_mat[m].std(axis=1, ddof=1)/np.sqrt(num_trials)
        mean_f1   = f1_mat[m].mean(axis=1);   se_f1   = f1_mat[m].std(axis=1, ddof=1)/np.sqrt(num_trials)
        out[m] = dict(mean_frac=mean_frac, se_frac=se_frac, mean_f1=mean_f1, se_f1=se_f1)
    return out



def frac_f1_over_bx(bx_array, n, p, s, snr, eps,
                    Lambda_ridge=100, tolr=5e-3, iters_pgd=5000, perc1mist=.1,
                    R=None, num_trials=10, Lambda=20.0, rho=0.1, sigma=0.1, r=1.1,
                    seed_num=-1, num_iterations=100, num_chains=None,
                    l1bound=3.5, mcmc_iter=5000):
    if num_chains is None:
        num_chains = num_iterations

    methods = ['topk', 'mistakes', 'lasso', 'mcmc']
    frac_mat = {m: np.zeros((len(bx_array), num_trials)) for m in methods}
    f1_mat   = {m: np.zeros((len(bx_array), num_trials)) for m in methods}

    for j, b_x in enumerate(bx_array):
        t_done = 0
        while t_done < num_trials:
            res = run_single_instance(
                n, p, s, snr, eps,
                Lambda_ridge=Lambda_ridge, tolr=tolr, iters_pgd=iters_pgd, perc1mist=perc1mist, Lambda=Lambda,
                seed_num=seed_num, R=R, rho=rho, sigma=sigma, r=r,
                b_x=b_x, b_y=b_x,
                num_iterations=num_iterations, 
                l1bound=l1bound, mcmc_iter=mcmc_iter,
                train_test=False, util_loss=False, frac_eps=1.0
            )
            sel = res.get('selection_full', None)
            if sel is None:
                continue

            beta_true01 = res['beta_true']
            supports = {
                'topk':     sel['supports_topk'],
                'mistakes': sel['supports_mistakes'],
                'lasso':    sel['supports_lasso'],
                'mcmc':     sel['supports_mcmc'],
            }
            for m in methods:
                frac, f1m = _scores_from_draws(beta_true01, supports[m], average=True)
                frac_mat[m][j, t_done] = frac
                f1_mat[m][j, t_done]   = f1m
            t_done += 1

    out = dict()
    for m in methods:
        mean_frac = frac_mat[m].mean(axis=1); se_frac = frac_mat[m].std(axis=1, ddof=1) / np.sqrt(num_trials)
        mean_f1   = f1_mat[m].mean(axis=1);   se_f1   = f1_mat[m].std(axis=1, ddof=1)   / np.sqrt(num_trials)
        out[m] = dict(mean_frac=mean_frac, se_frac=se_frac, mean_f1=mean_f1, se_f1=se_f1)
    return out


def util_loss_over_n(n_array, p, s, snr, eps,
                     Lambda_ridge=100, tolr=5e-3, iters_pgd=5000, perc1mist=.1,
                     R=None, num_trials=10, Lambda=20.0, rho=0.1, sigma=0.1, r=1.1,
                     b_x=1.0, b_y=1.0, seed_num=-1, num_iterations=100, num_chains=None,
                     l1bound=3.5, mcmc_iter=5000, frac_eps=0.5):
    """
    Returns per-method mean±SE of Obj-Pert utility-loss gaps over n.
    """
    if num_chains is None:
        num_chains = num_iterations

    methods = ['topk', 'mistakes', 'lasso', 'mcmc']
    gap_mat = {m: np.zeros((len(n_array), num_trials)) for m in methods}

    for j, n in enumerate(n_array):
        t_done = 0
        while t_done < num_trials:
            res = run_single_instance(
                n, p, s, snr, eps,
                Lambda_ridge=Lambda_ridge, tolr=tolr, iters_pgd=iters_pgd, perc1mist=perc1mist, Lambda=Lambda,
                seed_num=seed_num, R=R, rho=rho, sigma=sigma, r=r,
                b_x=b_x, b_y=b_y,
                num_iterations=num_iterations, 
                l1bound=l1bound, mcmc_iter=mcmc_iter,
                train_test=False, util_loss=True, frac_eps=frac_eps
            )
            UL = res.get('util_loss', None)
            if UL is None:
                continue
            # average over draws within the trial
            gap_mat['mistakes'][j, t_done] = np.mean(UL['gap_mistakes'])
            gap_mat['topk'][j, t_done]     = np.mean(UL['gap_topk'])
            gap_mat['lasso'][j, t_done]    = np.mean(UL['gap_lasso'])
            gap_mat['mcmc'][j, t_done]     = np.mean(UL['gap_mcmc'])
            t_done += 1

    out = dict()
    for m in methods:
        mean_gap = gap_mat[m].mean(axis=1)
        se_gap   = gap_mat[m].std(axis=1, ddof=1) / np.sqrt(num_trials)
        out[m] = dict(mean_gap=mean_gap, se_gap=se_gap)
    return out


def test_mse_over_n(n_array, p, s, snr, eps,
                    Lambda_ridge=100, tolr=5e-3, iters_pgd=5000, perc1mist=.1,
                    R=None, num_trials=10, Lambda=20.0, rho=0.1, sigma=0.1, r=1.1,
                    b_x=1.0, b_y=1.0, seed_num=-1, num_iterations=100, num_chains=None,
                    l1bound=3.5, mcmc_iter=5000, frac_eps=0.5):
    """
    Returns per-method mean±SE of **test MSE** over n (Obj-Pert on TRAIN, eval on TEST).
    """
    if num_chains is None:
        num_chains = num_iterations

    methods = ['topk', 'mistakes', 'lasso', 'mcmc']
    mse_mat = {m: np.zeros((len(n_array), num_trials)) for m in methods}

    for j, n in enumerate(n_array):
        t_done = 0
        while t_done < num_trials:
            res = run_single_instance(
                n, p, s, snr, eps,
                Lambda_ridge=Lambda_ridge, tolr=tolr, iters_pgd=iters_pgd, perc1mist=perc1mist, Lambda=Lambda,
                seed_num=seed_num, R=R, rho=rho, sigma=sigma, r=r,
                b_x=b_x, b_y=b_y,
                num_iterations=num_iterations, 
                l1bound=l1bound, mcmc_iter=mcmc_iter,
                train_test=True, util_loss=False, frac_eps=frac_eps
            )
            TM = res.get('test_mse', None)
            if TM is None:
                continue
            # average over draws within the trial
            mse_mat['mistakes'][j, t_done] = np.mean(TM['mse_mistakes'])
            mse_mat['topk'][j, t_done]     = np.mean(TM['mse_topk'])
            mse_mat['lasso'][j, t_done]    = np.mean(TM['mse_lasso'])
            mse_mat['mcmc'][j, t_done]     = np.mean(TM['mse_mcmc'])
            t_done += 1

    out = dict()
    for m in methods:
        mean_mse = mse_mat[m].mean(axis=1)
        se_mse   = mse_mat[m].std(axis=1, ddof=1) / np.sqrt(num_trials)
        out[m] = dict(mean_mse=mean_mse, se_mse=se_mse)
    return out

# =========================
# experiment: sweep over n
# =========================
eps0 = 1
p0 = 100
s0 = 5
mcmc_iterations = 1000
lambda_penalty = 120
perc1mistakes = 0.1
snr0 = 5
rho0 = 0.1
n_array = np.arange(3000, 8001, 1000)  # Number of rows
num_trial = 10
num_draws_per_trial = 100    # T: number of support draws per method per trial
frac_eps_util_mse = 0.5      # split ε for util-loss / test-MSE runs
clip_constant = 0.5

start = time.time()

# ---------- 1) variable-selection metrics (fraction-correct & F1) ----------
vs_stats = frac_f1_over_n(
    n_array, p=p0, s=s0, snr=snr0, eps=eps0,
    Lambda_ridge=lambda_penalty, tolr=5e-3, iters_pgd=5000, perc1mist=perc1mistakes,
    R=None, num_trials=num_trial, Lambda=20.0, rho=rho0, sigma=0.1, r=1.1,
    b_x=clip_constant, b_y=clip_constant, seed_num=-1, num_iterations=num_draws_per_trial,
    l1bound=3.5, mcmc_iter=mcmc_iterations
)

# ---------- 2) utility-loss (Obj-Pert gaps on full data) ----------
ul_stats = util_loss_over_n(
    n_array, p=p0, s=s0, snr=snr0, eps=eps0*2,
    Lambda_ridge=lambda_penalty, tolr=5e-3, iters_pgd=5000, perc1mist=perc1mistakes,
    R=None, num_trials=num_trial, Lambda=20.0, rho=rho0, sigma=0.1, r=1.1,
    b_x=clip_constant, b_y=clip_constant, seed_num=-1, num_iterations=num_draws_per_trial,
    l1bound=3.5, mcmc_iter=mcmc_iterations, frac_eps=frac_eps_util_mse
)

# ---------- 3) test-MSE (Obj-Pert β on TRAIN, eval on TEST) ----------
tm_stats = test_mse_over_n(
    n_array, p=p0, s=s0, snr=snr0, eps=eps0*2,
    Lambda_ridge=lambda_penalty, tolr=5e-3, iters_pgd=5000, perc1mist=perc1mistakes,
    R=None, num_trials=num_trial, Lambda=20.0, rho=rho0, sigma=0.1, r=1.1,
    b_x=clip_constant, b_y=clip_constant, seed_num=-1, num_iterations=num_draws_per_trial,
    l1bound=3.5, mcmc_iter=mcmc_iterations, frac_eps=frac_eps_util_mse
)

print(f"total runtime: {time.time() - start:.1f}s")

num_colors = 4
hue_values = np.linspace(0, 1, num_colors, endpoint=False)
colors_rgb = [hsv_to_rgb([hue, 1, 1]) for hue in hue_values]

labels = ['Top R', 'Mistakes', 'Lasso Subsample', 'MCMC']
order  = ['topk', 'mistakes', 'lasso', 'mcmc']  # aligns with colors_rgb[0..3]

def _tag():
    return f"p{p0}_s{s0}_l{lambda_penalty}_snr{snr0}_rho{rho0}_mcmc{mcmc_iterations}_eps{eps0}"

# ---------- Plot A: Fraction-correct ----------
plt.clf()
for i, key in enumerate(order):
    y  = vs_stats[key]['mean_frac']
    se = vs_stats[key]['se_frac']
    plt.errorbar(n_array, y, yerr=se, fmt='o-', capsize=5, capthick=1, elinewidth=1,
                 color=colors_rgb[i], label=labels[i])
plt.legend(fontsize=14)
ax = plt.gca()
ax.set_xticks(n_array) 
ax.tick_params(labelsize=14)
plt.yticks(fontsize=14)
plt.title(f'p={p0}', fontsize=16)
plt.xlabel('n', fontsize=14)
plt.ylabel('# correct supports / total trials', fontsize=14)
plt.grid(alpha=0.25, linestyle=':')
plt.savefig(f'./{_tag()}_frac_correct.png', dpi=200, bbox_inches='tight')

# ---------- Plot B: F1 score ----------
plt.clf()
for i, key in enumerate(order):
    y  = vs_stats[key]['mean_f1']
    se = vs_stats[key]['se_f1']
    plt.errorbar(n_array, y, yerr=se, fmt='o-', capsize=5, capthick=1, elinewidth=1,
                 color=colors_rgb[i], label=labels[i])
plt.legend(fontsize=14)
ax = plt.gca()
ax.set_xticks(n_array) 
ax.tick_params(labelsize=14)
plt.yticks(fontsize=14)
plt.title(f'p={p0}', fontsize=16)
plt.xlabel('n', fontsize=14)
plt.ylabel('F1 score', fontsize=14)
plt.grid(alpha=0.25, linestyle=':')
plt.savefig(f'./{_tag()}_f1.png', dpi=200, bbox_inches='tight')

# ---------- Plot C: Utility loss (Obj-Pert gap) ----------
plt.clf()
for i, key in enumerate(order):
    y  = ul_stats[key]['mean_gap']
    se = ul_stats[key]['se_gap']
    plt.errorbar(n_array, y, yerr=se, fmt='o-', capsize=5, capthick=1, elinewidth=1,
                 color=colors_rgb[i], label=labels[i])
plt.legend(fontsize=14)
ax = plt.gca()
ax.set_xticks(n_array) 
ax.tick_params(labelsize=14)
plt.yticks(fontsize=14)
plt.title(f'p={p0}', fontsize=16)
plt.xlabel('n', fontsize=14)
plt.ylabel('Obj-Pert utility loss', fontsize=14)
plt.grid(alpha=0.25, linestyle=':')
plt.savefig(f'./{_tag()}_util_loss.png', dpi=200, bbox_inches='tight')

# ---------- Plot D: Test MSE ----------
plt.clf()
for i, key in enumerate(order):
    y  = tm_stats[key]['mean_mse']
    se = tm_stats[key]['se_mse']
    plt.errorbar(n_array, y, yerr=se, fmt='o-', capsize=5, capthick=1, elinewidth=1,
                 color=colors_rgb[i], label=labels[i])
plt.legend(fontsize=14)
ax = plt.gca()
ax.set_xticks(n_array) 
ax.tick_params(labelsize=14)
plt.yticks(fontsize=14)
plt.title(f'p={p0}', fontsize=16)
plt.xlabel('n', fontsize=14)
plt.ylabel('Test MSE', fontsize=14)
plt.grid(alpha=0.25, linestyle=':')
plt.savefig(f'./{_tag()}_test_mse.png', dpi=200, bbox_inches='tight')

# ## ABLATION STUDIES

TOPR_COLOR = colors_rgb[0]   # red
MIST_COLOR = colors_rgb[1]   # light green

# --- Sweeps ---
bx_array     = np.array([0.5, 0.75, 1.0, 1.25, 1.50])        # edit as needed
lambda_array = np.array([120,140,160,180,200])             # ridge λ values
n_fixed      = 4000                # use a middle n

# --- Stats over b_x ---
bx_stats = frac_f1_over_bx(
    bx_array, n=n_fixed, p=p0, s=s0, snr=snr0, eps=eps0,
    Lambda_ridge=300, tolr=5e-3, iters_pgd=5000, perc1mist=perc1mistakes,
    R=None, num_trials=num_trial, Lambda=20.0, rho=rho0, sigma=0.1, r=1.1,
    seed_num=-1, num_iterations=num_draws_per_trial,
    l1bound=3.5, mcmc_iter=mcmc_iterations
)

# --- Stats over lambda (ridge) ---
lam_stats = frac_f1_over_lambda(
    lambda_array, n=n_fixed, p=p0, s=s0, snr=snr0, eps=eps0,
    tolr=5e-3, iters_pgd=5000, perc1mist=perc1mistakes,
    R=None, num_trials=num_trial, Lambda=20.0, rho=rho0, sigma=0.1, r=1.1,
    b_x=clip_constant, b_y=clip_constant, seed_num=-1, num_iterations=num_draws_per_trial,
    l1bound=3.5, mcmc_iter=mcmc_iterations
)

# ---- Fraction-correct vs b_x ----
plt.clf()
plt.errorbar(bx_array, bx_stats['topk']['mean_frac'],     yerr=bx_stats['topk']['se_frac'],
             fmt='o-', capsize=5, capthick=1, elinewidth=1, color=TOPR_COLOR,  label='Top R')
plt.errorbar(bx_array, bx_stats['mistakes']['mean_frac'], yerr=bx_stats['mistakes']['se_frac'],
             fmt='s--', capsize=5, capthick=1, elinewidth=1, color=MIST_COLOR, label='Mistakes')
ax = plt.gca(); ax.set_xticks(bx_array); ax.tick_params(labelsize=14)
plt.yticks(fontsize=14)
plt.legend(fontsize=14)
plt.xlabel('$b_x$', fontsize=14); plt.ylabel('# correct supports / total trials', fontsize=14)
plt.grid(alpha=0.25, linestyle=':'); plt.savefig(f'./{_tag()}_ablate_bx_frac.png', dpi=200, bbox_inches='tight')

# ---- F1 vs b_x ----
plt.clf()
plt.errorbar(bx_array, bx_stats['topk']['mean_f1'],     yerr=bx_stats['topk']['se_f1'],
             fmt='o-', capsize=5, capthick=1, elinewidth=1, color=TOPR_COLOR,  label='Top R')
plt.errorbar(bx_array, bx_stats['mistakes']['mean_f1'], yerr=bx_stats['mistakes']['se_f1'],
             fmt='s--', capsize=5, capthick=1, elinewidth=1, color=MIST_COLOR, label='Mistakes')
ax = plt.gca(); ax.set_xticks(bx_array); ax.tick_params(labelsize=14)
plt.yticks(fontsize=14)
plt.legend(fontsize=14)
plt.xlabel('$b_x$', fontsize=14); plt.ylabel('f1 score', fontsize=14)
plt.grid(alpha=0.25, linestyle=':'); plt.savefig(f'./{_tag()}_ablate_bx_f1.png', dpi=200, bbox_inches='tight')

# ---- Fraction-correct vs lambda ----
plt.clf()
plt.errorbar(lambda_array, lam_stats['topk']['mean_frac'],     yerr=lam_stats['topk']['se_frac'],
             fmt='o-', capsize=5, capthick=1, elinewidth=1, color=TOPR_COLOR,  label='Top R')
plt.errorbar(lambda_array, lam_stats['mistakes']['mean_frac'], yerr=lam_stats['mistakes']['se_frac'],
             fmt='s--', capsize=5, capthick=1, elinewidth=1, color=MIST_COLOR, label='Mistakes')
ax = plt.gca(); ax.set_xticks(lambda_array); ax.tick_params(labelsize=14)
plt.yticks(fontsize=14)
plt.legend(fontsize=14)
plt.xlabel('λ', fontsize=14); plt.ylabel('# correct supports / total trials', fontsize=14)
plt.grid(alpha=0.25, linestyle=':'); plt.savefig(f'./{_tag()}_ablate_lambda_frac.png', dpi=200, bbox_inches='tight')

# ---- F1 vs lambda ----
plt.clf()
plt.errorbar(lambda_array, lam_stats['topk']['mean_f1'],     yerr=lam_stats['topk']['se_f1'],
             fmt='o-', capsize=5, capthick=1, elinewidth=1, color=TOPR_COLOR,  label='Top R')
plt.errorbar(lambda_array, lam_stats['mistakes']['mean_f1'], yerr=lam_stats['mistakes']['se_f1'],
             fmt='s--', capsize=5, capthick=1, elinewidth=1, color=MIST_COLOR, label='Mistakes')
ax = plt.gca(); ax.set_xticks(lambda_array); ax.tick_params(labelsize=14)
plt.yticks(fontsize=14)
plt.legend(fontsize=14)
plt.xlabel('λ', fontsize=14); plt.ylabel('f1 score', fontsize=14)
plt.grid(alpha=0.25, linestyle=':'); plt.savefig(f'./{_tag()}_ablate_lambda_f1.png', dpi=200, bbox_inches='tight')

# Use small (p, s) so brute force is feasible
p_bf, s_bf = 100, 3
R_array = np.array([2, 300, *range(1000, 14001, 1000), 14200, 14300, 15000])
assert R_array.max() <= math.comb(p_bf, s_bf)

R_stats = frac_f1_over_R(
    R_array, n=1000, p=p_bf, s=s_bf, snr=snr0, eps=eps0,
    Lambda_ridge=80, tolr=5e-3, iters_pgd=5000, perc1mist=perc1mistakes,
    num_trials=10, Lambda=20.0, rho=rho0, sigma=0.1, r=1.1,
    b_x=clip_constant, b_y=clip_constant, seed_num=-1,
    num_iterations=num_draws_per_trial, l1bound=3.5, mcmc_iter=mcmc_iterations
)

# Single figure, two Top-R curves (same red; different markers/linestyles)
plt.clf()
plt.errorbar(R_array, R_stats['topk']['mean_f1'],   yerr=R_stats['topk']['se_f1'],
             fmt='o-',  capsize=5, capthick=1, elinewidth=1, color=TOPR_COLOR, label='F1')
plt.errorbar(R_array, R_stats['topk']['mean_frac'], yerr=R_stats['topk']['se_frac'],
             fmt='s--', capsize=5, capthick=1, elinewidth=1, color='k', label='Fraction correct', alpha=0.8)
ax = plt.gca()
ax.set_xticks(np.arange(0, 15001, 3000))
ax.tick_params(labelsize=14)
plt.yticks(fontsize=14)
# mark bucket boundaries
plt.axvline(1+291, color='k', lw=1, alpha=0.3)                 # end of ≤1 mistake
plt.axvline(1+291+13968, color='k', lw=1, alpha=0.3)           # end of ≤2 mistakes
plt.legend(fontsize=14)
plt.xlabel('R', fontsize=14)
plt.grid(alpha=0.25, linestyle=':'); plt.savefig(f'./p{p_bf}_s{s_bf}_topR_vs_R.png', dpi=200, bbox_inches='tight')