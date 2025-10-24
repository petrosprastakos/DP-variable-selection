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
mpmath.mp.dps = 100


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

    epsilon = np.random.normal(loc=0, scale=sigma, size=n)

    alpha = np.linalg.norm(X @ beta_true, ord=2)/(snr*np.linalg.norm(epsilon, ord=2))

    z = X @ beta_true + alpha*epsilon
    
    p = 1 / (1 + np.exp(-z))

    y = np.where(np.random.rand(n) < p, 1, -1)

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

    # Initialize the Gurobi model
    env = gp.Env(empty=True)
    env.setParam("OutputFlag",0)
    env.start()
    model = gp.Model("mip1", env=env)

    # Decision variables
    beta = model.addVars(p, lb=-GRB.INFINITY, name="beta")  # No bounds on beta
    theta = model.addMVar(p, name="theta")
    xi = model.addVars(n, lb=0.0, name="xi")  # Slack variables, xi >= 0
    z = model.addVars(p, vtype=GRB.BINARY, name="z")  # Binary variables for L0 constraint

    # Objective: (1/n) * sum(xi_i)
    model.setObjective((1 / n) * gp.quicksum(xi[i] for i in range(n))+ Lambda/(n)*gp.quicksum(theta[i] for i in range(p)), GRB.MINIMIZE)

    # Hinge loss constraints: xi_i >= 1 - y_i * (x_i^T * beta)
    for i in range(n):
        model.addConstr(
            xi[i] >= 1 - y[i] * gp.quicksum(X[i, j] * beta[j] for j in range(p)),
            name=f"hinge_constraint_{i}"
        )

    # Add constraints |βi| <= θi * zi
    for i in range(p):
        model.addConstr(beta[i]*beta[i] <= theta[i] * z[i])

    # Add constraint sum(z) <= s
    model.addConstr(gp.quicksum(z[i] for i in range(p)) <= s)
    model.addConstr(gp.quicksum(theta[i] for i in range(p)) <= l*l)


    # Optimize the model
    model.optimize()

    # Get the optimal solution
    beta = np.array([beta[j].X for j in range(p)])

    return beta


def lasso_gurobi(X, y, Lambda=20.0):
    
    n, p = X.shape
    
    # Initialize the Gurobi model

    env = gp.Env(empty=True)
    env.setParam("OutputFlag",0)
    env.start()
    model = gp.Model("mip1", env=env)

    # Decision variables
    beta = model.addVars(p, lb=-GRB.INFINITY, name="beta")  # No bounds on beta
    xi = model.addVars(n, lb=0.0, name="xi")  # Slack variables, xi >= 0
    z = model.addVars(p, lb=0.0, name="z")

    # Objective: (1/n) * sum(xi_i)
    model.setObjective((1 / n) * gp.quicksum(xi[i] for i in range(n))+ Lambda/n * gp.quicksum( z[j] for j in range(p)), GRB.MINIMIZE)

    # Hinge loss constraints: xi_i >= 1 - y_i * (x_i^T * beta)
    for i in range(n):
        model.addConstr(
            xi[i] >= 1 - y[i] * gp.quicksum(X[i, j] * beta[j] for j in range(p)),
            name=f"hinge_constraint_{i}"
        )

    for i in range(p):
            model.addConstr(beta[i] <= z[i])
            model.addConstr(-z[i] <= beta[i])
    


    # Optimize the model
    model.optimize()


    # Get the optimal solution
    obj_value = model.objVal
    if model.status == GRB.OPTIMAL:
        beta = np.array([beta[j].X for j in range(p)])
        return beta
    else:
        print("Optimization failed. Status code:", model.status)
        return None

def split_matrix(X, y, num_blocks):
    
    n, p = X.shape
    base_rows_per_block = n // num_blocks  # Minimum rows per block
    extra_rows = n % num_blocks            # Extra rows to distribute
    
    # Initialize the list for rows per block
    rows_per_block = [base_rows_per_block] * num_blocks
    
    # Distribute the extra rows across the first few blocks
    for i in range(extra_rows):
        rows_per_block[i] += 1

    # Create blocks
    blocks_X = []
    blocks_y = []
    start_idx = 0
    for rows_in_block in rows_per_block:
        blocks_X.append(X[start_idx:start_idx + rows_in_block, :])
        blocks_y.append(y[start_idx:start_idx + rows_in_block])
        start_idx += rows_in_block

    return blocks_X, blocks_y


def subsample(X, y, s, eps,func,factor = 1, num_iterations=50, Lambda=-1):
    
    n,p = X.shape
    blocks_X, blocks_y = split_matrix(X,y, factor*int(math.sqrt(n)))
    num_blocks = len(blocks_X)
    theta = np.zeros((num_blocks,p))
    V_matrix = np.zeros((num_blocks,p)) 
    Lambda_init = Lambda

    for i in range(0,num_blocks):
        if Lambda_init != -1:
            #we are doing lasso subsampling
            theta_fw = func(blocks_X[i], blocks_y[i], Lambda_init)
            num_nonzeros = np.count_nonzero(np.abs(theta_fw) > 1e-8)

            max_iters = 100
            iters = 0
            Lambda_low, Lambda_high = 1.0, Lambda_init
            while num_nonzeros != s and (Lambda_high - Lambda_low) > 5e-1 and iters < max_iters:
                iters += 1
                Lambda = (Lambda_high + Lambda_low) / 2
                theta_fw = func(blocks_X[i], blocks_y[i], Lambda)
                num_nonzeros = np.count_nonzero(np.abs(theta_fw) > 1e-8)
                if num_nonzeros > s:
                    Lambda_low = Lambda
                else:
                    Lambda_high = Lambda
            theta[i,:]= theta_fw
            sorted_indices = np.argsort(-np.abs(theta_fw))
            V_matrix[i,sorted_indices[:s]] = 1

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


def hinge_loss_backsolve(X, y, beta0, r=1.1, Lambda=10e-2, max_iter = 1000):
    n = X.shape[0]
    obj_new = np.zeros((max_iter,))
    beta = beta0 

    for i in range(max_iter):
        # Compute the gradient
        Xb0 = X @ beta
        Xb = y* Xb0
        yX = -X * y[:, np.newaxis]

        indices = np.where(Xb < 1)[0]
        grad = 1/n*(np.sum(yX[indices,:], axis=0)) + 2*Lambda/(n)* (beta)

        # Perform the gradient update
        beta_new = beta - 1/np.sqrt(i+1) * grad

        # Project onto the l2 ball
        beta_new = project_l2_ball(beta_new,r)

        # obj = obj_new
        obj_new[i] = 1/(n)*(np.sum(np.maximum(1-Xb, 0))) + Lambda/(n)*np.sum(beta_new**2)
        if i>0 and abs(obj_new[i] - obj_new[i-1])/obj_new[i] < 1e-8: #if objective values are starting to converge, end the PGD
            beta = beta_new
            obj_new = obj_new[0:i]
            # print(f"i exited pgd after {i} iterations")
            break
        beta = beta_new
    # print(f"i exited pgd after {i} iterations")
    return beta, obj_new[-1]



def project_l2_ball(x, r):
    return r/max(r,np.linalg.norm(x))*x


def compute_c_and_gradc_pgd(X, y, z, r, beta0, Lambda, max_iter = 1000):
    n = X.shape[0]
    obj_new = np.zeros((max_iter,))
    beta = project_l2_ball(beta0,r) ## project just in case initial point is infeasible
    yX = -X * y[:, None]

    for i in range(max_iter):
        # Compute the gradient
        Xb0 = X @ beta
        Xb = y* Xb0

        mask = (Xb < 1)
        grad = 1/n*(yX[mask].sum(axis=0)) + 2*Lambda/(n)* (beta/z)

        # Perform the gradient update
        beta_new = beta - 1/np.sqrt(i+1) * grad

        # Project onto the l2 ball
        beta_new = project_l2_ball(beta_new,r)

        obj_new[i] = 1/(n)*(np.sum(np.maximum(1-Xb, 0))) + Lambda/(n)*np.sum(beta_new**2 / z)
        if i>0 and abs(obj_new[i] - obj_new[i-1])/obj_new[i] < 1e-8: #if objective values are starting to converge, end the PGD
            beta = beta_new
            obj_new = obj_new[0:i]
            # print(f"i exited pgd after {i} iterations")
            break
        beta = beta_new
    # print(f"i exited pgd after {i} iterations")
    c = obj_new[-1]
    gradc = -Lambda/(n)*np.square(beta)/np.square(z)

    return beta, c, gradc


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
    # model = gp.Model()
    z = model.addMVar(p, vtype=GRB.BINARY, name="z")
    ita = model.addVar(lb=0.0, name="ita")

    model.setObjective(ita, GRB.MINIMIZE)

    zt= np.zeros((p,))
    zt[(p-s):p] = 1     #last s entries of s1 are 1
    # print(f"the current zt is {zt }")

    b = X.T @ y

    

    zt_hat = add_noise(zt)
    # print(f"the zt after heuristic is {zt}")
    beta0, c,gradc = compute_c_and_gradc_pgd(X, y, zt_hat, r, zt, Lambda, iter_pgd)
    gradcT = np.transpose(gradc)
    model.addConstr(c + gradcT @ (z - zt_hat) <= ita)

    # print(f"the current zt is {zt }, the #iter is {i}")
    ita_t = 0
    t = 0
    # print(f"the current cost is {c } and the current gradient is {gradc}")
    one_indices = np.nonzero(zt)[0]
    Xs = X[:, one_indices]
    beta_s0 = beta0[one_indices]
    beta_s, curr_obj = hinge_loss_backsolve(Xs, y, beta_s0, r, Lambda)

    model.addConstr(gp.quicksum(z[i] for i in range(p)) <= s)
    while np.abs(ita_t -curr_obj)/curr_obj > 1e-3:
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
        beta_warm, c, gradc = compute_c_and_gradc_pgd(X, y, zt_hat, r, beta0, Lambda, iter_pgd)
        # a = zt_hat
        beta0 = beta_warm
        gradcT = np.transpose(gradc)
        model.addConstr(c + gradcT @ (z - zt_hat) <= ita)
        one_indices = np.nonzero(zt)[0]
        Xs = X[:, one_indices]
        beta_s0 = beta0[one_indices]
        beta_s, curr_obj = hinge_loss_backsolve(Xs, y, beta_s0, r, Lambda)
        # print(f"the current cost is {c } and the current gradient is {gradc}")
    print(f"best: {zt}")
    print(f"iterations first set: {t}")
    zt[np.argsort(np.abs(zt))[:p-s]] = 0 #fixes precision errors


    one_indices_opt = np.nonzero(zt)[0]

    # Find indices where beta_true is 0
    zero_indices = np.where(zt == 0)[0]


    corr_indices = np.argsort(np.abs(b))[::-1]  # Sort indices in descending order of absolute values
    idx_end = int(per1mist * len(b))  # Calculate the index corresponding to the top 10%
    top_10_corr_indices = corr_indices[:idx_end]  # Get the indices of top 10% elements

    idx_zeros = zero_indices[np.isin(zero_indices, top_10_corr_indices)]
    idx_ones = one_indices_opt[np.isin(one_indices_opt, top_10_corr_indices)]

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
                    _, c, gradc = compute_c_and_gradc_pgd(X, y, mistake_noise, r, beta0, Lambda, iter_pgd)
                    # print(f"pgd took {time.time()-bfpgd}")
                    bfpgd = time.time()
                    gradcT = np.transpose(gradc)
                    model.addConstr(c + gradcT @ (z - mistake_noise) <= ita)
                # print(f"adding cut took {time.time()-bfpgd}")
        if k== 1:
            break

    betas = np.zeros((p,(p-s)*s+2))
    betas_mistakes = np.zeros((p,s+1))
    betas[:,0] = zt
    betas_mistakes[:,0] = zt

    gaps_mistakes = np.zeros((s+1,))
    gaps_mistakes[0]  = model.getAttr(GRB.Attr.MIPGap)

    obj_values_mistakes = np.zeros((s+1,))
    obj_values = np.zeros(((p-s)*s+2,))
    Xs = X[:, one_indices]
    beta_s0 = beta0[one_indices]

    _, obj_values[0] = hinge_loss_backsolve(Xs, y, beta_s0, r, Lambda)
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

            beta_s, obj_values[k]= hinge_loss_backsolve(Xs, y, beta_s0, r, Lambda)
            beta_s0 = beta_s
            k +=1
    # print(f"backsolving took {time.time()-bfbacksolve}")


    model.addConstr(gp.quicksum(z[i] for i in one_indices_opt) <= s-1.5)

    ita_t = 0
    t=0
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
        beta_warm, c, gradc = compute_c_and_gradc_pgd(X, y, zt_hat, r, beta0, Lambda, iter_pgd)
        beta0 = beta_warm
        gradcT = np.transpose(gradc)
        model.addConstr(c + gradcT @ (z - zt_hat) <= ita)
        one_indices = np.nonzero(zt)[0]
        Xs = X[:, one_indices]
        beta_s0 = beta0[one_indices]
        beta_s, curr_obj = hinge_loss_backsolve(Xs, y,beta_s0, r, Lambda)
    print(f"best with 2 mistakes: {zt}")
    print(f"iterations second set: {t}")
    zt[np.argsort(np.abs(zt))[:p-s]] = 0 #fixes precision errors
    betas[:,1+(p-s)*s] = zt
    nonzero_indices = np.nonzero(zt)[0]
    Xs = X[:, nonzero_indices]
    beta_s, obj_values[1+(p-s)*s] = hinge_loss_backsolve(Xs, y,beta_s0, r, Lambda)
    

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
            beta_warm, c, gradc = compute_c_and_gradc_pgd(X, y, zt_hat, r, beta0, Lambda, iter_pgd)
            beta0 = beta_warm
            gradcT = np.transpose(gradc)
            model.addConstr(c + gradcT @ (z - zt_hat) <= ita)
            one_indices = np.nonzero(zt)[0]
            Xs = X[:, one_indices]
            beta_s0 = beta0[one_indices]
            beta_s, curr_obj = hinge_loss_backsolve(Xs, y,beta_s0, r, Lambda)
        print(f"best with {j+1} mistakes: {zt}")
        print(f"iterations {j+1} set: {t}")
        zt[np.argsort(np.abs(zt))[:p-s]] = 0 #fixes precision errors
        betas_mistakes[:,j+1] = zt
        nonzero_indices = np.nonzero(zt)[0]
        Xs = X[:, nonzero_indices]
        beta_s, obj_values_mistakes[j+1] = hinge_loss_backsolve(Xs, y,beta_s0, r, Lambda)

    return sorted_betas, gap, sorted_obj_values, gaps_mistakes, betas_mistakes, obj_values_mistakes


def add_noise(zt, lb=1e-3, ub=5e-3):
    zero_indices = np.where(zt == 0)[0]
    zt_hat = zt.copy()
    zt_hat[zero_indices] += np.random.uniform(lb, ub, size=len(zero_indices))
    return zt_hat

def run_single_instance(n, p, s, snr, eps, Lambda_ridge = 100,tolr=5e-3, iters_pgd=5000, perc1mist=.1, Lambda=20.0, seed_num = -1, R=100, rho = 0.1, sigma = 0.1,  r = 1.1, pool_num = 900, b_x = 1, b_y = 1, num_iterations = 50, num_chains = 50, l1bound = 3.5, block_mult = 1):

    if (seed_num == -1):
        np.random.seed() 
    else:
        np.random.seed(seed_num)
    
    #compute sensitivity
    delta = 1/(n)*(1+b_x*r*np.sqrt(s))

    X, y, beta_true = generate_input_data(n, p, sigma, s, rho,snr)

    X_new = np.where(np.abs(X) > b_x, np.sign(X) * b_x, X)
    start = time.time()
    betas, gap, obj_values, gaps_mistakes, betas_mistakes, obj_values_mistakes = outer_approx_gurobi_constraint_pgd_with_cuts(y, X_new, s, r, Lambda_ridge, tol=tolr, iter_pgd=iters_pgd, per1mist=perc1mist)
    print(f"OA took {time.time()-start} seconds")

    if gap == -1:
        print("i got here")
        return -1, -1, -1, -1, -1, -1, -1, -1, -1
    
    R = (p-s)*s+2
    reduced_matrix = betas

    start = time.time()
    support_lasso = subsample(X, y, s, eps, lasso_gurobi, block_mult, num_iterations, Lambda)
    print(f"Lasso took {time.time()-start} seconds")

    idx = find_matching_column(beta_true, reduced_matrix, R-1)


    exponents = np.zeros((R+1,))
    exponents[0:R] = -eps*obj_values/(2*delta)
    exponents[R] = -eps*obj_values[R-1]/(2*delta)

    exp_terms = np.array([mpmath.exp(x) for x in exponents])
    exp_terms[R] = (math.comb(p,s) - R)*exp_terms[R]
    denom = mpmath.fsum(exp_terms)
    prob = exp_terms/denom
    prob = prob.astype(float)

    exponents = np.zeros((s+1,))
    exponents = -eps*obj_values_mistakes/(2*delta)
    exp_terms = np.array([mpmath.exp(x) for x in exponents])
    for i in range(0,s+1):
        comb_term = math.comb(p-s,i)*math.comb(s,i)
        exp_terms[i] = comb_term*exp_terms[i]
    denom = mpmath.fsum(exp_terms)
    prob_mistakes = exp_terms/denom
    prob_mistakes = prob_mistakes.astype(float)
    prob2 = prob_mistakes

    return obj_values, reduced_matrix, beta_true, prob, idx,support_lasso, prob2, betas_mistakes, obj_values_mistakes

def varying_parameters(n_array, p, s, snr, eps, Lambda_ridge=100, tolr=5e-3, iters_pgd=5000, perc1mist=.1, R=100, num_trials = 10, Lambda = 20.0,num_iterations=100, rho = 0.1, sigma = 0.1,  r = 1.1, pool_num = 900, b_x = 1, b_y = 1, seed_num = -1, num_chains = 50, l1bound = 3.5, block_mult = 1):
    R = (p-s)*s+2
    lasso_correct_pr = np.zeros((len(n_array),num_trials))
    f1_lasso = np.zeros((len(n_array),num_trials))

    topk_correct = np.zeros((len(n_array),num_trials))
    f1_topk = np.zeros((len(n_array),num_trials))

    mistakes_correct = np.zeros((len(n_array),num_trials))
    f1_mistakes = np.zeros((len(n_array),num_trials))
    
    for j in range(0,len(n_array)):
        k = 0
        while k <= (num_trials -1):
            obj_values, reduced_matrix, beta_true, prob, idx, support_lasso, prob2, betas_mistakes, obj_values_mistakes = run_single_instance(n_array[j], p, s, snr, eps, Lambda_ridge, tolr, iters_pgd, perc1mist, Lambda, seed_num, R, rho, sigma, r, pool_num, b_x,b_y, num_iterations, num_chains, l1bound, block_mult)
            if idx == -1:
                continue
            prob = np.cumsum(prob)
            prob2 = np.cumsum(prob2)
            beta_true = np.where(np.array(beta_true) != 0, 1, 0)

            for i in range(num_iterations):
                reduced_matrix = np.where(np.array(reduced_matrix) != 0, 1, 0)
                index = draw_index(prob, np.random.uniform(0, 1))
                topk_beta = np.zeros((p,))
                if index != R:
                    topk_beta = reduced_matrix[:, index]
                else: #we are in the k > R scenario
                    flag = 0
                    while flag == 0:
                        shuffled_nums = np.arange(p)
                        np.random.shuffle(shuffled_nums)
                        nonzero_indices = shuffled_nums[0:s]
                        topk_beta = np.zeros((p,))
                        topk_beta[nonzero_indices] = 1

                        for l in range(0,R):
                            if np.array_equal(reduced_matrix[:,l], topk_beta):
                                break
                            if l == R-1:
                                flag = 1

                index2 = draw_index(prob2, np.random.uniform(0, 1))
                mistake_beta = betas_mistakes[:,index2]

                mistake_beta = np.where(np.array(mistake_beta) != 0, 1, 0)

                f1_lasso[j,k] += f1_score(beta_true, support_lasso[:,i])/num_iterations
                f1_topk[j,k] += f1_score(beta_true, topk_beta)/num_iterations
                f1_mistakes[j,k] += f1_score(beta_true, mistake_beta)/num_iterations

                if np.array_equal(support_lasso[:,i], beta_true):
                    lasso_correct_pr[j,k] += 1/num_iterations
                if np.array_equal(topk_beta, beta_true):
                    topk_correct[j,k] += 1/num_iterations
                if np.array_equal(mistake_beta, beta_true):
                    mistakes_correct[j,k] += 1/num_iterations
            k += 1
            
    return topk_correct, f1_topk, mistakes_correct, f1_mistakes, lasso_correct_pr, f1_lasso


num_colors = 3
hue_values = np.linspace(0, 1, num_colors, endpoint=False)

# Convert HSV colors to RGB
colors_rgb = [hsv_to_rgb([hue, 1, 1]) for hue in hue_values]


eps0 = 1
p0 = 1000
s0 = 5
lambda_penalty = 100
perc1mistakes = 0.01
snr0 = 5
rho0 = 0.1
n_array = np.arange(3000, 6001, 1000)  # Number of rows
num_trial = 10
start = time.time()

print(f"Here are the solns to n={n_array}, p={p0},s={s0}, rho={rho0}, snr={snr0}, lambda={lambda_penalty}, eps={eps0}:")

topk_correct, f1_topk, mistakes_correct, f1_mistakes, lasso_correct_pr, f1_lasso = varying_parameters(n_array, p=p0, s=s0,  rho=rho0, snr=snr0, eps=eps0, b_x = 0.5, b_y = 0.5, num_trials = num_trial, Lambda_ridge=lambda_penalty, tolr=5e-3, iters_pgd=5000, perc1mist=perc1mistakes)
print(f"This took {time.time()-start} seconds")
print(f"Here are the solns to n={n_array}, p={p0},s={s0}, rho={rho0}, snr={snr0}, lambda={lambda_penalty}, eps={eps0}:")
print(topk_correct, f1_topk, mistakes_correct, f1_mistakes,lasso_correct_pr, f1_lasso)

topk_correct_mean = np.mean(topk_correct,axis = 1)
mistakes_correct_mean = np.mean(mistakes_correct,axis = 1)
lasso_correct_pr_mean = np.mean(lasso_correct_pr,axis = 1)

f1_topk_mean = np.mean(f1_topk,axis = 1)
f1_mistakes_mean = np.mean(f1_mistakes,axis = 1)
f1_lasso_mean = np.mean(f1_lasso,axis = 1)

print(topk_correct_mean, f1_topk_mean, mistakes_correct_mean, f1_mistakes_mean, lasso_correct_pr_mean, f1_lasso_mean)

topk_correct_se = np.std(topk_correct, axis=1) / np.sqrt(topk_correct.shape[1])
mistakes_correct_se = np.std(mistakes_correct, axis=1) / np.sqrt(mistakes_correct.shape[1])
lasso_correct_pr_se = np.std(lasso_correct_pr, axis=1) / np.sqrt(lasso_correct_pr.shape[1])

f1_topk_se = np.std(f1_topk, axis=1) / np.sqrt(f1_topk.shape[1])
f1_mistakes_se = np.std(f1_mistakes, axis=1) / np.sqrt(f1_mistakes.shape[1])
f1_lasso_se = np.std(f1_lasso, axis=1) / np.sqrt(f1_lasso.shape[1])

# Plotting
plt.clf()
plt.errorbar(n_array, topk_correct_mean, yerr=topk_correct_se, fmt='o-', capsize=5, capthick=1, elinewidth=1, color=colors_rgb[0], label='Top R')
plt.errorbar(n_array, mistakes_correct_mean, yerr=mistakes_correct_se, fmt='o-', capsize=5, capthick=1, elinewidth=1, color=colors_rgb[1], label='Mistakes')
plt.errorbar(n_array, lasso_correct_pr_mean, yerr=lasso_correct_pr_se, fmt='o-', capsize=5, capthick=1, elinewidth=1, color=colors_rgb[2], label='Lasso Subsample')


# Add legend
plt.legend(fontsize=14)

plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.title(f'p={p0}', fontsize=16)

# Add labels and title
plt.xlabel('n', fontsize=14)
plt.ylabel('# correct supports/total trials', fontsize=14)
plt.savefig(f'./hinge_p{p0}_s{s0}_l{lambda_penalty}_snr{snr0}_rho{rho0}_eps{eps0}.png', format='png')

plt.clf()
plt.errorbar(n_array, f1_topk_mean, yerr=f1_topk_se, fmt='o-', capsize=5, capthick=1, elinewidth=1, color=colors_rgb[0], label='Top R')
plt.errorbar(n_array, f1_mistakes_mean, yerr=f1_mistakes_se, fmt='o-', capsize=5, capthick=1, elinewidth=1, color=colors_rgb[1], label='Mistakes')
plt.errorbar(n_array, f1_lasso_mean, yerr=f1_lasso_se, fmt='o-', capsize=5, capthick=1, elinewidth=1, color=colors_rgb[2], label='Lasso Subsample')


# Add legend
plt.legend(fontsize=14)

plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.title(f'p={p0}', fontsize=16)

# Add labels and title
plt.xlabel('n', fontsize=14)
plt.ylabel('f1 score', fontsize=14)
plt.savefig(f'./hinge_f1_p{p0}_s{s0}_l{lambda_penalty}_snr{snr0}_rho{rho0}_eps{eps0}.png', format='png')