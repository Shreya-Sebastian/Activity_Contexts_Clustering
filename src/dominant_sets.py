import numpy as np
import math

# functions to execute dominant sets clustering algorithm
# http://homepage.tudelft.nl/3e2t5/HungKrose_ICMI2011.pdf
def symmetrize_A(A, op):
    if op == 'max':
        W = np.maximum( A, A.transpose() )
    elif op== 'min':
        W = np.minimum(A, A.transpose())
    elif op =='avg':
        W = (A + A.transpose())/2
    return W

# d-sets function k
def k(S, i, A):
    sum_affs = 0
    for j in range(len(S)):
        if S[j]:
            sum_affs += A[i,j]

    return 1/np.sum(S) * sum_affs

# d-sets function phi
def phi(S,i,j,A):
    return A[i,j] - k(S,i,A)

# d-sets function weight
def weight(S, i, A):
    if np.sum(S) == 1:
        return 1
    else:
        R = S.copy()
        R[i] = False
        sum_weights = 0
        for j in range(len(R)):
            if R[j]:
                sum_weights += phi(R,j,i,A) * weight(R, j, A)
        return sum_weights

## optimization function
def f(x, A):
    return np.dot(x.T, np.dot(A, x))

# Thesis Sec 5.2.3: "runs to a fixed tolerance of 10^-6 under a 1000-iteration
# cap; in practice convergence is reached in far fewer than 1000 iterations
# and the cap is a safeguard." Both values are cited/documented there.
CONVERGENCE_TOL = 1e-6
MAX_CLIMB_ITERS = 1000

## iteratively finds vector x which maximizes f
def vector_climb(A, allowed, num_nodes, original_A, thres=1e-5):

    x = np.full(num_nodes, 1.0 / num_nodes)
    x = np.multiply(x, allowed)
    eps = 10
    n = 10
    n_iter = 0
    while eps > CONVERGENCE_TOL and n_iter < MAX_CLIMB_ITERS:
        p = f(x,A)
        x = np.multiply(x, np.dot(A,x)) / np.dot(x, np.dot(A,x))
        n = f(x,A)
        eps = abs(n-p)
        n_iter += 1

    groups = x > thres
 
    for i in range(num_nodes):
        if not allowed[i]:
            if weight(groups,i,original_A) > 0.0:
                return []
    return groups

# Finds vectors x of people which maximize f. Then removes those people and repeats
def iterate_climb_learned(A, num_nodes):
    allowed = np.ones(num_nodes)
    groups = []

    original_A = A.copy()
    while (np.sum(allowed) > 1):
        A[allowed == False] = 0
        A[:,allowed == False] = 0
        if (np.sum(np.dot(allowed,A)) == 0):
            break
        x = vector_climb(A, allowed, num_nodes, original_A, thres=1e-5)
        # len(x) == 0 only catches vector_climb's explicit `[]` return; a
        # converged-but-all-below-threshold result is a boolean array of
        # length num_nodes with no True entries, which len() does not
        # detect -- without checking np.any(x) too, `allowed` never shrinks
        # and this loop spins forever on that case.
        if len(x) == 0 or not np.any(x):
            break
        groups.append(x)
        allowed = np.multiply(x == False, allowed)
    return groups
    
def dominant_set_extraction(A,num_nodes,op='min'):
    A_= symmetrize_A(A, op)
    # print("symmetric A:", A_)
    groups = iterate_climb_learned(A_,num_nodes)
    return groups

