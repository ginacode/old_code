import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import progressbar  # TODO this should be tqdm
from train import train_overdl
from tqdm import tqdm

# TODO clean up entirely


# ----------------------- ARGUMENTS --------------------------
parser = argparse.ArgumentParser(description='Dictionary learning over various sizes')
parser.add_argument('exp_name', type=str,
                    help='name of experiment')
parser.add_argument('--learning_method', type=str,
                    help='MOD1, MOD1_omp, spams, alt_prox_proj')
parser.add_argument('--n_true_atoms', type=int, default=50,
                    help='number of ground truth atoms')
parser.add_argument('--n_features', type=int, default=20,
                    help='number of features')
parser.add_argument('--k_sparsity', type=int, default=3,
                    help='sparsity (max number of nonzero elements)')
parser.add_argument('--dict_sizes', nargs='+', type=int)
args = parser.parse_args()


# ----------------------- PARAMETERS ------------------------
exp_name = args.exp_name

# dictionary & data parameters
n_components = args.n_true_atoms
n_features = args.n_features
sparsity = args.k_sparsity

# learning parameters
train_size = 200
test_size = 1000 # needs to be the same size, as long as eval on X_test is done with specific code

dl_method = args.learning_method
coding_method_train = 'OMP'  # 'OMP' or 'L1' or 'L1_2'
coding_method_eval = 'OMP'  # 'OMP' or 'L1' or 'L1_2'
num_iter = 1000
num_epoches = 1
n_trials = int(5)


# ------------- EXPERIMENT AS A FUNCTION OF DICTIONARY SIZE ---------------
dict_size = args.dict_sizes
dict_metric_0 = dict.fromkeys(['distance', 'err_k1', 'err_k3'])
risk_metric_0 = dict.fromkeys(['train', 'test', 'test_OMP', 'oracle'])
dict_metric_prnd = dict.fromkeys(['distance', 'err_k1', 'err_k3'])
risk_metric_prnd = dict.fromkeys(['train', 'test', 'test_OMP', 'oracle'])

for key in dict_metric_0:
    dict_metric_0[key] = np.zeros((len(dict_size), n_trials))
    dict_metric_prnd[key] = np.zeros((len(dict_size), n_trials))
for key in risk_metric_0:
    risk_metric_0[key] = np.zeros((len(dict_size), n_trials))
    risk_metric_prnd[key] = np.zeros((len(dict_size), n_trials))

for i, d_size in enumerate(tqdm(dict_size[0:]), start=0):
    d_metric, r_metric, d_metric_prnd, r_metric_prnd \
        = train_overdl(d_size, dl_method, coding_method_train, coding_method_eval, n_trials,
                       n_components, n_features, sparsity, train_size, test_size,
                       num_iter, num_epoches, nnz_init='Gaussian', clean=False)
    for key in dict_metric_0:
        dict_metric_0[key][i, :] = d_metric[key]
        dict_metric_prnd[key][i, :] = d_metric_prnd[key]
    for key in risk_metric_0:
        risk_metric_0[key][i, :] = r_metric[key]
        risk_metric_prnd[key][i, :] = r_metric_prnd[key]


# ---------------------------- SAVE DATA --------------------------------
os.makedirs(f'{exp_name}', exist_ok=True)
np.save(f'{exp_name}/dict_sizes.npy', np.array(dict_size))
np.save(f'{exp_name}/risk_metric_train.npy', risk_metric_0['train'])
np.save(f'{exp_name}/risk_metric_test.npy', risk_metric_0['test'])
np.save(f'{exp_name}/risk_metric_test_OMP.npy', risk_metric_0['test_OMP'])
np.save(f'{exp_name}/dict_distance.npy', dict_metric_0['distance'])

# ---------------------------- PLOTTING ---------------------------------
plt.style.use('bmh')


def plot_stats(x, Y, color='', label=''):
    plt.semilogx(x, np.mean(Y, axis=1), label=label)
    # plt.fill_between(x, np.mean(Y,axis=1) - np.std(Y,axis=1), np.mean(Y,axis=1) + np.std(Y,axis=1),
    # color=color, alpha=0.2)
    plt.fill_between(x, np.percentile(Y, 25, axis=1), np.percentile(Y, 75, axis=1), alpha=0.2)

# --------------------------------------------------------
fig, ax1 = plt.subplots()

color = 'tab:blue'
ax1.set_xlabel('p\'')
ax1.set_ylabel('Training error', color=color)
ax1.semilogx(dict_size, np.mean(risk_metric_0['train'], axis=1), color=color)
ax1.fill_between(dict_size, np.percentile(risk_metric_0['train'], 25, axis=1), np.percentile(risk_metric_0['train'], 75, axis=1), alpha=0.2, color=color)
ax1.tick_params(axis='y', labelcolor=color)

ax1 = plt.gca()
ax1.set_xlim(dict_size[0], dict_size[-1])
ax1.set_xscale('log')

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

color = 'tab:red'
ax2.set_ylabel('Test error', color=color)
ax2.semilogx(dict_size, np.mean(risk_metric_0['test'], axis=1), color=color)
ax2.fill_between(dict_size, np.percentile(risk_metric_0['test'], 25, axis=1), np.percentile(risk_metric_0['test'], 75, axis=1), alpha=0.2, color=color)
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.savefig(f'{exp_name}/risk.pdf', transparent=True)

# --------------------------------------------------------
fig, ax1 = plt.subplots()

color = 'tab:blue'
ax1.set_xlabel('p\'')
ax1.set_ylabel('Training error', color=color)
ax1.semilogx(dict_size, np.mean(risk_metric_0['train'], axis=1), color=color)
ax1.fill_between(dict_size, np.percentile(risk_metric_0['train'], 25, axis=1), np.percentile(risk_metric_0['train'], 75, axis=1), alpha=0.2, color=color)
ax1.tick_params(axis='y', labelcolor=color)

ax1 = plt.gca()
ax1.set_xlim(dict_size[0], dict_size[-1])
ax1.set_xscale('log')

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

color = 'tab:red'
ax2.set_ylabel('Test error (OMP)', color=color)
ax2.semilogx(dict_size, np.mean(risk_metric_0['test_OMP'], axis=1), color=color)
ax2.fill_between(dict_size, np.percentile(risk_metric_0['test_OMP'], 25, axis=1), np.percentile(risk_metric_0['test_OMP'], 75, axis=1), alpha=0.2, color=color)
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.savefig(f'{exp_name}/risk_OMP.pdf', transparent=True)
# --------------------------------------------------------

#plt.show()

# # Test and training error
# fig = plt.figure()
# plot_stats(dict_size, risk_metric_0['train'], 'blue', label='train error')
# plot_stats(dict_size, risk_metric_0['test'], 'red', label='test error')
# # plot_stats(dict_size ,risk_metric_prnd['test'], 'blue', label='Test_pruned')
# # plot_stats(dict_size, risk_metric_prnd['train'], 'black', label='Train_pruned')
# # plot_stats(dict_size, risk_metric_0['oracle'], 'green', label='Oracle')
# ax1 = plt.gca()
# ax1.set_xlim(dict_size[0], dict_size[-1])
# ax1.set_xscale('log')
#
# plt.legend(loc='best')
# plt.xlabel('p\' ')
# plt.title('Risk')


# Dictionary distance
fig = plt.figure()
plot_stats(dict_size, dict_metric_0['distance'], 'red', label='Dictionary distance')
ax1 = plt.gca()
ax1.set_xlim(dict_size[0], dict_size[-1])
ax1.set_xscale('log')

plt.legend(loc='best')
plt.xlabel('p\' ')
plt.title('Recovery Error')
plt.gcf().subplots_adjust(bottom=0.1)
plt.savefig(f'{exp_name}/dict_distance.pdf', transparent=True)
