import numpy as np
from sklearn.metrics import pairwise_distances
from sklearn.decomposition import sparse_encode
from scipy.optimize import linear_sum_assignment
import spams

"""
Functions from dicteval.py, with minor modifications.
eval_objective: swapped X and D, changed structure
eval_dict_distance: removed extraneous condition
Modified all functions to work with X, D, R of consistent dimensions
"""

# TODO
# make the other functions "mine"


def eval_objective(X, D, sparsity, lambd, package='spams', coding_method='L1',
                   return_regularization=False, specific_code=None):
    """ Computes the objective value of the following optimization problem
        min (0.5 || X - D * code ||_F^2 + lambd * || code ||_1) / n_samples
        code

    Args:
        X: array of shape (n_features, n_samples)
            Data matrix.
        D: array of shape (n_features, n_components),
            Dictionary matrix
        sparsity:
            Sparsity for spams.omp coding method
        lambd:
            Sparsity controlling parameter.
        package: string that is 'spams' or not 'spams'
            'spams': uses the lasso lars implemented in the spams package
            not 'spams': uses sparse_encode from sklearn
        coding_method: package=spams, {'L1','OMP','L1_2'}; else {‘lasso_lars’, ‘lasso_cd’, ‘lars’, ‘omp’, ‘threshold’}
            coding method to be used, specified with spams or sklearn.sparse_encode argument string
        return_regularization (bool):
            if True, objective value computed with regularization added; otherwise, without
        specific_code: array of shape (n_components, n_samples)
            Sparse code matrix

    Returns:
        The objective value as a float.
    """

    if specific_code is None:
        assert package in ('spams', 'sklearn'), 'package argument needs to be "spams" or "sklearn".'
        # ---------------------- compute code from data and dictionary ----------------------
        if package == 'spams':
            if coding_method == 'L1':
                param = {'mode': 2, 'lambda1': lambd}
                code = spams.lasso(np.asfortranarray(X), D=np.asfortranarray(D), **param)
            elif coding_method == 'L1_2':
                param = {'mode': 0, 'lambda1': lambd, 'numThreads': 4}
                code = spams.lasso(np.asfortranarray(X), D=np.asfortranarray(D), **param)
            elif coding_method == 'OMP':
                code = spams.omp(np.asfortranarray(X), D=np.asfortranarray(D), L=sparsity)
        elif package == 'sklearn':
            code = sparse_encode(X.T, D.T, algorithm=coding_method, alpha=lambd, max_iter=5000)
    else:
        code = specific_code
    obj = 0.5 * np.linalg.norm(X - D @ code, ord='fro')**2

    if return_regularization:
        obj += lambd*np.linalg.norm(code.flatten(), 1)

    return obj / X.shape[1]


def eval_dict_distance(D1, D2, assignment='nearest_neighbor'):
    """Computes distance between two dictionaries.

    For assignment='permutation' (requires n_components_1=n_components_2), this computes
        min (||D1 - D2 * P||_F^2) / n_components_1
         P
    where P is the set of signed permutation matrices
    (see https://en.wikipedia.org/wiki/Generalized_permutation_matrix#Signed_permutation_group)
    For assignment='nearest_neighbor', this computes
        min (||D1 - D2 * P||_F^2) / n_components_1
         P
    where each column of P satisfies ||P_i||_0 = ||P_i||_1 = 1, i.e., has only one nonzero
    entry with value either 1 or -1.

    Note that for the distance to make sense, atoms in the dictionaries should be normalized

    Args:
        D1 : array of shape (n_features, n_components_1),
            Dictionary matrix
        D2 : array of shape (n_features, n_components_2),
            Second dictionary matrix
        assignment : {'permutation', 'nearest_neighbor'}
            method for assigning columns of D1 to columns of D2

    Returns:
        distance between dictionaries, as described above
    """

    pdist_p = pairwise_distances(D1.T, D2.T)
    pdist_n = pairwise_distances(D1.T, -D2.T)
    pdist = np.minimum(pdist_p, pdist_n)

    if assignment == 'permutation':
        assert (D1.shape[1] == D2.shape[1])
        row_ind, col_ind = linear_sum_assignment(pdist)
        distance = pdist[row_ind, col_ind].sum()

    elif assignment == 'nearest_neighbor':
        distance = np.sum(np.amin(pdist, axis=1))

    distance = distance / D1.shape[1]
    return distance


def compute_dict_k_distance(D1, D2, k, t=100):
    z = np.zeros((t, D1.shape[1]))

    for i in range(t):
        S = np.random.permutation(D1.shape[1])[0:k]
        z[i, S] = np.ones(k)  # np.random.randn(k)
    X = np.matmul(z, D1.T)

    code = spams.omp(np.asfortranarray(X.T), D=np.asfortranarray(D2), L=k)
    Xd = np.asfortranarray(X.T) - np.asfortranarray(D2) * code
    error = np.linalg.norm(Xd, 'fro') / np.sqrt(t) / k
    return error
