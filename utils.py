import numpy as np
from sklearn.preprocessing import normalize

"""
soft_thr from dictlearning_prun.py
get_Grass_matrix from train_overparameterized.py
"""
# TODO clean up entirely
# TODO should i be k-thresholding by magnitude? right now i just do it by value


def soft_thr(X, thresh):
    """ Decreases magnitude of all entries in X by thresh. If a magnitude is already
        less than thresh, then sets it to 0.0

        Args:
            X (np.ndarray): asd
    """
    return np.sign(X) * np.maximum(np.abs(X) - thresh, 0.0)


def k_thresholding(X, k):
    """ Selects k highest values from each column of an array X
        https://stackoverflow.com/a/37933092 """
    # Sort X, store the argsort for later usage
    sidx = np.argsort(X, axis=0)
    sA = X[sidx, np.arange(X.shape[1])]

    # Perform differentiation along rows and look for non-zero differentiations
    df = np.diff(sA, axis=0) != 0

    # Perform cumulative summation along rows from bottom upwards.
    # Thus, summations < K should give us a mask of valid ones that are to
    # be kept per column. Use this mask to set rest as zeros in sorted array.
    mask = (df[::-1].cumsum(0) < k)[::-1]
    sA[:-1] *= mask

    # Finally revert back to unsorted order by using sorted indices sidx
    out = sA[sidx.argsort(0), np.arange(sA.shape[1])]

    return out


def get_Grass_matrix(n,m):
    # construct an (approximate) Grassmannian matrix (maximally decorrelated atoms)
    delt = .95
    D = np.random.normal(size=(n,m))
    D = normalize(D.T).T
    res = 1
    Mus = np.array([1])
    while res > 0.00001:
        G = D.T.dot(D)
        max_mu = np.max(np.abs(G-np.eye(m)))
        G[np.abs(G-np.eye(m)) > max_mu*delt] = max_mu *delt * np.sign(G[np.abs(G-np.eye(m)) > max_mu*delt])
        U, S, V = np.linalg.svd(G)
        D = np.diag(np.sqrt(S[0:n])).dot(U[:,:n].T)
        D = normalize(D.T).T
        res = Mus[-1]-max_mu
        Mus = np.append(Mus,max_mu)
    return D