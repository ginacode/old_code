import numpy as np
from sklearn.preprocessing import normalize
import spams

from dict_learn import dict_learning
from dict_eval import eval_objective, eval_dict_distance, compute_dict_k_distance

"""
From dictlearning_prun.py, with minor modifications
"""
# TODO clean up entirely


class Dictionary(object):
    """ Class for dictionary learning and evaluation
    """
    def __init__(self, dict_size, n_features, sparsity, coding_method):
        self.dict_size = dict_size
        self.n_features = n_features
        self.sparsity = sparsity
        self.reset_dict()

    def reset_dict(self):
        # Initialize or reset dictionary
        rdict = np.random.normal(size=(self.dict_size, self.n_features))
        self.rdict = normalize(rdict)

    def update_dict(self, X, lambd, method='spams', mode=2, num_iter=1000, clean=True, step_size=0.1, batch_size=100):
        # Update dictionary with training data
        if method == 'MOD1' or method == 'MOD_dict' or method == 'MOD_em' or method == 'MOD1_omp':
            self.rdict, self.rank_sparse_code = dict_learning(X.T, self.dict_size, lambd, method=method, mode=mode,
                                                              init_rdict=self.rdict.T, num_iter=num_iter, clean=clean,
                                                              step_size=step_size, batch_size=batch_size)
        else:
            self.rdict = dict_learning(X.T, self.dict_size, lambd, method=method, mode=mode, init_rdict=self.rdict.T,
                                       num_iter=num_iter, clean=clean, step_size=step_size, batch_size=batch_size)
        self.rdict = self.rdict.T

    def eval_dict(self, dictionary, X, X_test, lambd, coding_method, test_sparsity=''):
        dictionary = dictionary.T
        X = X.T
        X_test = X_test.T

        if test_sparsity == '': test_sparsity = self.sparsity

        # Evaluate recovered dictionary (i.e. r_dict) against true dictionary (i.e. dictionary)
        dict_metric = {'distance': eval_dict_distance(dictionary, self.rdict.T),
                       'err_k1': compute_dict_k_distance(dictionary, self.rdict.T, 1, t=1000),
                       'err_k3': compute_dict_k_distance(dictionary, self.rdict.T, 3, t=1000)}

        risk_metric = {'train': eval_objective(X, self.rdict.T, test_sparsity, lambd,
                                               package='spams', coding_method=coding_method,
                                               return_regularization=False),
                       'test': eval_objective(X_test, self.rdict.T, test_sparsity, lambd,
                                              package='spams', coding_method=coding_method,
                                              return_regularization=False),
                       'oracle': eval_objective(X_test, dictionary, test_sparsity, lambd,
                                                package='spams', coding_method=coding_method,
                                                return_regularization=False)}
        return dict_metric, risk_metric

    def prune_dict(self, m_target, X_test, sparsity, method):
        if method == "incoherence":
            D = self.rdict.T
            (n, m) = D.shape
            cnt = 0

            if m > m_target:
                G = np.abs(D.T.dot(D)) - np.eye(m)
                corrs = np.sum(G, axis=0)
                order = np.argsort(corrs)
                D = D[:, order[0:m_target]]
                m = D.shape[1]

            self.rdict = D.T
            self.dict_size = self.rdict.shape[0]

            G = D.T.dot(D)
            max_mu = np.max(np.abs(G - np.eye(m)))
            print("Final Size", self.dict_size, ", Mutual coherence: ", max_mu)

        elif method == "coding":
            if self.dict_size > m_target:
                # print('pruning with ', sparsity, 'atoms')
                code = spams.omp(np.asfortranarray(X_test.T), D=np.asfortranarray(self.rdict.T), L=sparsity)
                code[np.nonzero(code)] = 1
                uses = np.array(np.sum(code, axis=1))
                order = np.argsort(np.squeeze(uses))
                D = self.rdict.T
                self.rdict = D[:, order[-m_target:]].T
                self.dict_size = self.rdict.shape[0]

