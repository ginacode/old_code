import numpy as np
from sklearn.preprocessing import normalize
import sys
from dict import Dictionary

"""
From train_overparameterized.py, with minor modifications
"""
# TODO clean up entirely


def gen_data(n_components, n_features, sparsity, train_size, test_size, dict_init, nnz_init, noise_std=0):
    # generate dictionary
    if dict_init == 'Gaussian':
        dictionary = np.random.randn(n_components, n_features)
    elif dict_init == 'Grassmannian':
        dictionary = get_Grass_matrix(n_features, n_components).T
    dictionary = normalize(dictionary)

    # generate data
    N = train_size + test_size
    Z = np.zeros((n_components, N))
    for i in range(N):
        S = np.random.permutation(n_components)[0:sparsity]
        if nnz_init == 'Gaussian':
            Z[S, i] = np.random.randn(sparsity)
        elif nnz_init == 'ones':
            Z[S, i] = np.ones(sparsity)
        else:
            sys.exit("Incorrect nnz_init")

    D = np.random.randn(n_features, n_components)
    D0 = D.dot(np.diag(np.sqrt(np.diag(D.T.dot(D))) ** (-1)))
    X = np.asfortranarray(D0.dot(Z) + noise_std * np.random.randn(n_features, N), dtype=float)
    X = normalize(X, axis=0)
    dictionary = np.asfortranarray(D0, dtype=float).T
    X_train = X[:, 0:train_size].T
    X_test = X[:, train_size:].T

    return dictionary, X_train, X_test


def train_overdl(dict_size, dl_method, coding_method_train, coding_method_eval, n_trials, n_components, n_features,
                 sparsity, train_size, test_size, num_iter, num_epoches, dict_init='Gaussian', nnz_init='Gaussian',
                 clean=True, sparsity_prun='', sparsity_test='', batch_size=100, noise_std=0):
    # Train dictionary of size dict_size and return metrics on it and on the pruned version (and the last resulting dictionary)

    if sparsity_prun == '': sparsity_prun = sparsity

    if sparsity_test == '': sparsity_test = sparsity

    if coding_method_train == 'OMP':
        mode = 3
        lambd = sparsity
    elif coding_method_train == 'L1':
        mode = 2
        # lambd = 0.15 * (num_iter*(9/4000) + 5/4) * (train_size/2250 + 5/9) # * dict_size / 70
        lambd = .1
    elif coding_method_train == 'L1_2':
        if nnz_init == 'Gaussian': sys.exit("Incorrect Data Method for this mode")
        mode = 0
        lambd = sparsity
    else:
        sys.exit("Incorrect Coding method")

    # Metrics for overcomplete Dict
    dict_metric = dict.fromkeys(['distance', 'err_k1', 'err_k3'])
    risk_metric = dict.fromkeys(['train', 'test', 'test_OMP', 'oracle'])

    # Metrics for Dict after Pruning
    dict_metric_prnd = dict.fromkeys(['distance', 'err_k1', 'err_k3'])
    risk_metric_prnd = dict.fromkeys(['train', 'test', 'test_OMP', 'oracle'])

    for key in dict_metric:
        dict_metric[key] = np.zeros(n_trials)
        dict_metric_prnd[key] = np.zeros(n_trials)
    for key in risk_metric:
        risk_metric[key] = np.zeros(n_trials)
        risk_metric_prnd[key] = np.zeros(n_trials)

    for trial in range(n_trials):
        # generate dictionary and data
        dictionary, X, X_test = gen_data(n_components, n_features, sparsity, train_size, test_size, dict_init, nnz_init,
                                         noise_std=noise_std)

        # learn dictionary
        model = Dictionary(dict_size, n_features, sparsity, coding_method_train)

        for epoch in range(num_epoches):
            model.update_dict(X, lambd, method=dl_method, mode=mode, num_iter=num_iter, clean=clean,
                              batch_size=batch_size)
        dict_raw = model.rdict.T
        d_metric, r_metric = model.eval_dict(dictionary, X, X_test, lambd, coding_method_eval, test_sparsity=sparsity_test,
                                             return_regularization=False, specific_code=model.paired_code)

        # Pruning
        model.prune_dict(n_components, X_test, sparsity_prun, 'coding')
        d_metric_prnd, r_metric_prnd = model.eval_dict(dictionary, X, X_test, lambd, coding_method_eval, test_sparsity=sparsity_test,
                                                       return_regularization=False, specific_code=None)  # TODO prune_dict needs to modify specific_code

        for key in dict_metric:
            dict_metric[key][trial] = d_metric[key]
            dict_metric_prnd[key][trial] = d_metric_prnd[key]
        for key in risk_metric:
            risk_metric[key][trial] = r_metric[key]
            risk_metric_prnd[key][trial] = r_metric_prnd[key]

    return dict_metric, risk_metric, dict_metric_prnd, risk_metric_prnd