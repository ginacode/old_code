import numpy as np
from sklearn.preprocessing import normalize
import spams
from utils import k_thresholding, soft_thr
from scipy.sparse.linalg.eigen.arpack import eigsh

"""
From dictlearning_prun.py, with minor modifications
"""
# TODO: clean up entirely


def dict_learning(X, dict_size, lambd, method='spams', mode=2, init_rdict=None, num_iter=1000, clean=True,
                  step_size=0.1, batch_size=100, full_rank_only=True):
    """Solves a dictionary learning matrix factorization problem.--

    Finds the best dictionary and the corresponding sparse code for
    approximating the data matrix X by solving::
        min 0.5 || X - rdict * code ||_2^2 + lambd * || code ||_1
    (dictionary,code)
        with || rdict_k ||_2 = 1 for all 0 <= k < dict_size

    Parameters
    ----------
    X : array of shape (n_features, n_samples)
        Data matrix.
    dict_size : int,
        Number of dictionary atoms to extract.
    lambd :
        Sparsity controlling parameter.
    method : {'spams', 'alt_prox_proj'}
        spams: perform online dictionary learning implemented in SPAMS
        alt_prox_proj: iteratively update code via one step proximal GD and
        rdict via one step projected GD
    num_iter : int,
        Number of iterations to perform.
    step_size :
        Step size in alt_prox_proj
    """

    if init_rdict is None:
        rdict = np.random.normal(size=(X.shape[0], dict_size))
        rdict = normalize(rdict, axis=0)
    else:
        rdict = init_rdict

    if method == 'spams':
        param = {'mode': mode,
                 'K': dict_size,
                 'lambda1': lambd,
                 'batchsize': batch_size,
                 'iter': num_iter,
                 'verbose': False,
                 'D': rdict,
                 'clean': clean}

        if mode == 2: param['D'] = None
        if mode == 0: param['D'] = None

        rdict = spams.trainDL(np.asfortranarray(X), **param)
        # rdict = normalize(rdict, axis=0)  # make sure it is normalized

    elif method == 'alt_prox_proj':
        code = np.zeros((dict_size, X.shape[1]))

        for iter in range(num_iter):
            residual = (X - rdict @ code)
            code += step_size * rdict.T @ residual
            code = soft_thr(code, lambd * step_size)
            rdict += (0.2 * step_size) * residual @ code.T # 0.2 is empirically good
            rdict = normalize(rdict, axis=0)


    # -------------------------------------------------------------------
    # For the MOD1 methods, lambd refers to the sparsity-controlling parameter in the Basis Pursuit problem
    #     min 0.5 || X - rdict * code ||_F^2   s.t.  || rdict_k ||_0 <= lambd for all 0 <= k < dict_size
    # rather than for the LASSO problem mentioned above. This is because (1) we wish to see how the
    # dictionary update problem behaves as a function of the BP sparsity-controlling parameter and
    # (2) Gina is a lazy, messy coder.
    # --------------------------------------------------------------------
    elif method == 'MOD1':
        rdict = np.random.normal(size=(X.shape[0], dict_size))
        rdict = normalize(rdict, axis=0)

        # sparse coding step: "k-thresholding"
        code = rdict.T @ X
        code = k_thresholding(code, lambd)

        # keep initializing dictionary until full rank code is returned
        if code.shape[0] >= code.shape[1] and full_rank_only:
            while( np.linalg.matrix_rank(code) != code.shape[1] ):
                rdict = np.random.normal(size=(X.shape[0], dict_size))
                rdict = normalize(rdict, axis=0)

                # sparse coding step: "k-thresholding"
                code = rdict.T @ X
                code = k_thresholding(code, lambd)

        # dictionary update step
        rdict = X @ np.linalg.pinv(code)

        # normalize dictionary (by adjusting rows of code)
        rdict, dict_norm_coefficients = normalize(rdict, axis=0, return_norm=True)
        code = code * dict_norm_coefficients[:, None]

        return rdict, code

    elif method == 'MOD_dict':
        rdict = np.random.normal(size=(X.shape[0], dict_size))
        rdict = normalize(rdict, axis=0)

        # sparse coding step: "k-thresholding"
        code = rdict.T @ X
        code = k_thresholding(code, lambd)

        # keep initializing dictionary until full rank code is returned
        if code.shape[0] >= code.shape[1] and full_rank_only:
            while( np.linalg.matrix_rank(code) != code.shape[1] ):
                rdict = np.random.normal(size=(X.shape[0], dict_size))
                rdict = normalize(rdict, axis=0)

                # sparse coding step: "k-thresholding"
                code = rdict.T @ X
                code = k_thresholding(code, lambd)

        for _ in range(500):
            # dictionary gradient descent
            lr = 1/eigsh(code.T @ code, 1, which='LA', return_eigenvectors=False)[0]
            rdict = rdict - lr * ( -2.0 * (X - rdict @ code) @ code.T )

            # normalize dictionary (by adjusting rows of code)
            rdict, dict_norm_coefficients = normalize(rdict, axis=0, return_norm=True)
            code = code * dict_norm_coefficients[:, None]

            # should probably get a better stopping condition
            if np.linalg.norm(X - rdict @ code, ord='fro') <= 1e-12:
                print('break1')
                break

        return rdict, code

    elif method == 'MOD_em':
        rdict = np.random.normal(size=(X.shape[0], dict_size))
        rdict = normalize(rdict, axis=0)

        # sparse coding step: "k-thresholding"
        code = rdict.T @ X
        code = k_thresholding(code, lambd)

        # keep initializing dictionary until full rank code is returned
        if code.shape[0] >= code.shape[1] and full_rank_only:
            while( np.linalg.matrix_rank(code) != code.shape[1] ):
                rdict = np.random.normal(size=(X.shape[0], dict_size))
                rdict = normalize(rdict, axis=0)

                # sparse coding step: "k-thresholding"
                code = rdict.T @ X
                code = k_thresholding(code, lambd)

        # dictionary update step
        lr = 1/eigsh(code.T @ code, 1, which='LA', return_eigenvectors=False)[0]
        rdict = rdict - lr * ( -2.0 * (X - rdict @ code) @ code.T )

        # normalize dictionary
        rdict, dict_norm_coefficients = normalize(rdict, axis=0, return_norm=True)

        for iter in range(500):
            # sparse coding step: "k-thresholding"
            code = rdict.T @ X
            code = k_thresholding(code, lambd)

            # dictionary update step
            lr = 1 / eigsh(code.T @ code, 1, which='LA', return_eigenvectors=False)[0]
            rdict = rdict - lr * (-2.0 * (X - rdict @ code) @ code.T)

            # normalize dictionary (by adjusting rows of code)
            rdict, dict_norm_coefficients = normalize(rdict, axis=0, return_norm=True)
            code = code * dict_norm_coefficients[:, None]

            # should probably get a better stopping condition
            if np.linalg.norm(X - rdict @ code, ord='fro') <= 1e-12:
                print('break2')
                break

        return rdict, code

    elif method == 'MOD1_omp':
        for iter in range(1):
            # sparse coding step
            code = spams.omp(np.asfortranarray(X), np.asfortranarray(rdict), L=lambd).todense()

            # dictionary update step
            rdict = X @ np.linalg.pinv(code)
            rdict = normalize(rdict, axis=0)
        # https://stackoverflow.com/questions/54858314/rank-of-sparse-matrix-python
        return rdict, code

    return rdict, code