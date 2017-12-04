from itertools import permutations
from typing import Tuple, Dict
import warnings

import matplotlib as mpl
from matplotlib import pyplot as plt
# noinspection PyUnresolvedReferences
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Ellipse
import numpy as np
from scipy.cluster.hierarchy import dendrogram
from scipy.io import loadmat
from scipy.spatial.distance import cdist, pdist, squareform
from scipy.spatial import Voronoi, voronoi_plot_2d
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.metrics import roc_auc_score, classification_report


############################### Assignment 1 ##################################
# noinspection PyPep8Naming
def kmeans(X: np.ndarray, k: int, max_iter: int=100,
           print_progress=True) -> Tuple[np.ndarray, np.ndarray, float]:
    N, M = X.shape
    X = np.asarray(X)
    indexes = np.random.choice(np.arange(N), size=k, replace=False)
    mu = X[indexes].copy()
    r = np.zeros(N, dtype=int)
    last_r = np.zeros_like(r)
    loss_score = np.inf

    for i in range(max_iter):
        # assign all datapoints to their closest prototype
        dists = cdist(X, mu, metric='sqeuclidean')
        r = np.argpartition(dists, 0, axis=1)[:, 0]

        # re-compute the new positions of the prototypes
        # for this assignment
        loss_score = 0
        for j in range(k):
            X_assigned: np.ndarray = r == j
            # easiest way to handle empty clusters is to
            # just ignore them
            if X_assigned.any():
                cluster = X[X_assigned, :]
                mu[j] = cluster.mean(axis=0)
                loss_score += np.linalg.norm(cluster - mu[j])
            else:
                mu[j] = np.array([np.nan] * M)

        changes_nr = (last_r != r).sum()
        if print_progress:
            print(f"Iteration: {i}.")
            print(f"Number of changes: {changes_nr}.")
            print(f"Loss: {loss_score}")
        if changes_nr == 0:
            break
        last_r = r

    print(f"Number of iterations: {i + 1}. Loss: {loss_score}")

    return mu, r, loss_score
############################### Assignment 1 ##################################


############################### Assignment 2 ##################################
# noinspection PyPep8Naming
def loss(X, r):
    k = r.max() + 1
    loss_score = 0
    for i in range(k):
        cluster = X[r == i].copy()
        if len(cluster):
            mu = cluster.mean(axis=0)
            loss_score += np.linalg.norm(cluster - mu)
    return loss_score


# noinspection PyPep8Naming
def kmeans_agglo(X, r) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    N, M = X.shape
    k = r.max() + 1
    R = np.zeros((k - 1, N), dtype=int)
    kmloss = np.zeros(k)
    mergeidx = np.zeros((k - 1, 2), dtype=int)

    kmloss[0] = loss(X, r)
    for i in range(k - 1):
        old_r = r.copy()
        R[i] = old_r
        min_loss = np.inf
        for ri, rj in permutations(np.unique(old_r), 2):
            new_r = old_r.copy()
            new_r[new_r == ri] = rj
            new_loss = loss(X, new_r)
            if new_loss < min_loss:
                mergeidx[i, :] = ri, rj
                min_loss = new_loss
                r = new_r
        kmloss[i + 1] = min_loss
    return R, kmloss, mergeidx
############################### Assignment 2 ##################################


############################### Assignment 3 ##################################
# noinspection PyPep8Naming
def agglo_dendro(kmloss: np.ndarray, mergeidx: np.ndarray, ax=None) -> None:
    N, _ = mergeidx.shape
    Z = np.ones((N, 4))
    Z[:, :2] = mergeidx
    Z[:, 2] = kmloss[1:]
    max_r = mergeidx.max()
    for i, (_, c2) in enumerate(Z[:, :2]):
        max_r += 1
        offset = np.zeros(i+1, dtype=bool)
        mask = np.hstack((offset, Z[i+1:, 0] == c2))
        Z[mask, 0] = max_r
        mask = np.hstack((offset, Z[i+1:, 1] == c2))
        Z[mask, 1] = max_r

    show_fig = False
    if ax is None:
        plt.figure(figsize=(8, 8))
        ax = plt.gca()
        show_fig = True
    dendrogram(Z, ax=ax)
    if show_fig:
        plt.savefig('agglo_dendro.png')
        plt.show()
############################### Assignment 3 ##################################


############################### Assignment 4 ##################################
def get_nonsingular_C(C: np.ndarray, tol: float, modify_C: bool)\
        -> Tuple[np.ndarray, float]:
    _, M = C.shape
    det_C = np.linalg.det(C)
    i = 1
    while det_C < tol:
        if modify_C:
            C += np.eye(M) * tol * i
        else:
            C = C + np.eye(M) * tol * i
        det_C = np.linalg.det(C)
        i += 1
    return C, det_C


# noinspection PyPep8Naming
def norm_pdf(X: np.ndarray, mu: np.ndarray, C: np.ndarray, tol: float=1e-5,
             modify_C: bool=False) -> np.ndarray:
    _, M = X.shape
    C, det_C = get_nonsingular_C(C, tol, modify_C)
    X = X.T
    C = C.T
    mu = mu.reshape(M, 1)
    b = X - mu
    c, *_ = np.linalg.lstsq(C, b)
    denominator = (2*np.pi)**(M/2) * det_C**0.5
    nominator = np.diag(b.T @ c)
    nominator = np.exp(-nominator/2)
    return nominator/denominator
############################### Assignment 4 ##################################


############################### Assignment 5 ##################################
# noinspection PyPep8Naming
def em_gmm(X: np.ndarray, k: int, max_iter: int=100,
           init_kmeans: bool=False, tol: float=1e-5,
           print_progress=True)\
        -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    N, M = X.shape
    pi = np.ones(k)/k
    indexes = np.random.choice(np.arange(N), size=k, replace=False)
    if init_kmeans:
        mu, _, _ = kmeans(X, k, max_iter, print_progress=print_progress)
    else:
        mu = X[indexes].copy()
    sigma = np.zeros((k, M, M)) + np.eye(M)
    loglik = 0
    prev_loglik = 0
    gamma = np.zeros((k, N))
    for i in range(max_iter):
        # E-step
        for centr in range(k):
            gamma[centr] = pi[centr] * norm_pdf(X, mu[centr], sigma[centr],
                                                tol=tol, modify_C=True)
        gamma /= gamma.sum(axis=0)
        # M-step
        loglik = 0
        for centr in range(k):
            gamma_centr = gamma[centr, :].reshape(N, 1)
            Nk = gamma_centr.sum()
            pi[centr] = Nk/N
            mu[centr] = (X * gamma_centr).sum(axis=0)/Nk
            mu_centr = mu[centr]
            b = np.sqrt(gamma_centr) * (X - mu_centr)
            sigma[centr] = (b.T @ b)/Nk
            loglik += (gamma_centr * norm_pdf(X, mu_centr, sigma[centr],
                                              tol=tol, modify_C=True)).sum()
        loglik = np.log(loglik)
        if print_progress:
            print(f"Iteration: {i}. Log likelihood: {loglik}")
        if np.isclose(loglik, prev_loglik):
            break
        prev_loglik = loglik
    print(f"Number of iterations: {i + 1}. Log likelihood: {loglik}")
    return pi, mu, sigma, loglik
############################### Assignment 5 ##################################


############################### Assignment 6 ##################################
# noinspection PyPep8Naming
def plot_gmm_solution(X: np.ndarray, mu: np.ndarray,
                      sigma: np.ndarray, ax=None) -> None:
    show_fig = False
    if ax is None:
        plt.figure(figsize=(8, 8))
        ax = plt.gca()
        show_fig = True
    x_min, y_min = X.min(axis=0) - np.ones(2)
    x_max, y_max = X.max(axis=0) + np.ones(2)
    ax.scatter(*X.T)
    ax.scatter(*mu.T, marker='+', c='r')

    for sig, centr in zip(sigma, mu):
        lambda_, v = np.linalg.eig(sig)
        lambda_ = np.sqrt(lambda_)

        ell = Ellipse(xy=(centr[0], centr[1]),
                      width=lambda_[0], height=lambda_[1],
                      angle=np.rad2deg(np.arccos(v[0, 0])))
        ell.set_facecolor('none')
        ell.set_edgecolor('r')
        ell.set_linewidth(1)
        ax.add_artist(ell)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    if show_fig:
        plt.savefig('gmm.png')
        plt.show()


# noinspection PyPep8Naming
def plot_kmeans(X: np.ndarray, mu: np.ndarray, r: np.ndarray,
                ax=None) -> None:
    show_fig = False
    if ax is None:
        plt.figure(figsize=(8, 8))
        ax = plt.gca()
        show_fig = True

    x_min, y_min = X.min(axis=0) - np.ones(2)*0.25
    x_max, y_max = X.max(axis=0) + np.ones(2)*0.25

    for x, ri in zip(X, r):
        ax.text(*x, str(ri), alpha=0.75, color=sns.color_palette()[0])
    for i, m in enumerate(mu):
        ax.text(*m, str(i), fontweight='bold', fontsize=15, color='r')

    if mu.shape[0] > 2:
        vor = Voronoi(mu)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore',
                                  mpl.cbook.MatplotlibDeprecationWarning)
            voronoi_plot_2d(vor, ax=ax, show_points=False, line_colors='r')
    else:
        a, b = mu.mean(axis=0)
        m = -(mu[0, 0] - mu[1, 0])/(mu[0, 1] - mu[1, 1])
        ax.plot([x_min, 2*x_max], [(x_min - a)*m + b, (2*x_max - a)*m + b],
                '--r', lw=1)

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    if show_fig:
        plt.savefig('kmeans.png')
        plt.show()
############################### Assignment 6 ##################################


############################### Assignment 7 ##################################
def plot_clustering(data, axes, k, init_kmeans, tol=1e-5):
    print(f"k: {k}")
    km_mu, r, _ = kmeans(data, k, print_progress=False)
    _, kmloss, mergeidx = kmeans_agglo(data, r)
    gmm_pi, gmm_mu, gmm_sigma, _ = em_gmm(data, k, init_kmeans=init_kmeans,
                                          print_progress=False, tol=tol)
    if len(axes) == 3:
        ax1, ax2, ax3 = axes
        ax1.set_ylabel(fr"$k={k}$")
        ax1.set_xticks([])
        ax1.set_yticks([])
        ax2.set_yticks([])
        ax2.grid()
        ax3.set_xticks([])
        ax3.set_yticks([])
        agglo_dendro(kmloss, mergeidx, ax=ax2)
    else:
        ax1, ax3 = axes
        ax1.set_ylabel(fr"$k={k}$")
        ax1.set_xticks([])
        ax1.set_yticks([])
        ax3.set_xticks([])
        ax3.set_yticks([])
    plot_kmeans(data, km_mu, r, ax=ax1)
    plot_gmm_solution(data, gmm_mu, gmm_sigma, ax=ax3)
    return km_mu


def assignment7():
    gaussians5 = np.load('./5_gaussians.npy').T
    plt.figure()
    _, axes = plt.subplots(6, 3, figsize=(15, 5*6))
    for k, ax in enumerate(axes, 2):
        plot_clustering(gaussians5, ax, k, init_kmeans=False)
    plt.tight_layout()
    plt.savefig('assignment7_a.png')
    plt.show()

    plt.figure()
    _, axes = plt.subplots(6, 3, figsize=(15, 5*6))
    for k, ax in enumerate(axes, 2):
        plot_clustering(gaussians5, ax, k, init_kmeans=True)
    plt.tight_layout()
    plt.savefig('assignment7_b.png')
    plt.show()
############################### Assignment 7 ##################################


############################### Assignment 8 ##################################
def assignment8():
    gaussians2 = np.load('./2_gaussians.npy').T
    plt.figure()
    _, axes = plt.subplots(5, 3, figsize=(15, 5*5))
    k = 2
    for ax in axes:
        plot_clustering(gaussians2, ax, k, init_kmeans=False)
    plt.tight_layout()
    plt.savefig('assignment8_a.png')
    plt.show()

    plt.figure()
    _, axes = plt.subplots(5, 3, figsize=(15, 5*5))
    k = 2
    for ax in axes:
        plot_clustering(gaussians2, ax, k, init_kmeans=True)
    plt.tight_layout()
    plt.savefig('assignment8_b.png')
    plt.show()
############################### Assignment 8 ##################################


############################### Assignment 9 ##################################
# noinspection PyPep8Naming
def km_cluster_idx(usps: Dict[str, np.ndarray], r: np.ndarray) -> np.ndarray:
    # noinspection PyTypeChecker
    labels = np.argwhere(usps['data_labels'].T == 1)[:, 1]
    Y = usps['data_labels'].T
    y_pred = np.zeros_like(labels)
    for i in range(10):
        matching_label = Y[r == i].sum(axis=0).argmax()
        y_pred[r == i] = matching_label
    return y_pred


# noinspection PyPep8Naming
def gmm_cluster_idx(usps: Dict[str, np.ndarray], pi, mu, sigma, tol=1e-5):
    X = usps['data_patterns'].T
    N, _ = X.shape
    k, _ = mu.shape
    likelihood = np.zeros((N, k))

    for centr in range(k):
        likelihood[:, centr] = pi[centr] * norm_pdf(X, mu[centr], sigma[centr],
                                                    tol=tol) ** 2
    r = likelihood.argmax(axis=-1)

    return km_cluster_idx(usps, r)


# noinspection PyPep8Naming,PyUnresolvedReferences
def assignment9():
    usps = loadmat('./usps.mat')
    X = usps['data_patterns'].T
    plt.figure()
    _, axes = plt.subplots(3, 3, figsize=(15, 15))
    k = 10
    axes = axes.flatten()
    km_mu, r, _ = kmeans(X, k, print_progress=False)
    _, kmloss, mergeidx = kmeans_agglo(X, r)
    gmm_pi, gmm_mu, gmm_sigma, _ = em_gmm(X, k, init_kmeans=False,
                                          print_progress=False, tol=1e-1)
    ax1 = axes[0]
    ax1.grid()
    ax1.set_yticks([])
    agglo_dendro(kmloss, mergeidx, ax=ax1)
    for ax, (_, c2) in zip(axes[1:], mergeidx[1:]):
        ax.set_xticks([])
        ax.set_yticks([])
        ax.imshow(km_mu[c2].reshape(16, 16), interpolation='none',
                  aspect='equal')
    plt.tight_layout()
    plt.savefig('assignment9_1.png')
    plt.show()

    model = TSNE()
    data = np.vstack((X, km_mu, gmm_mu))
    proj = model.fit_transform(data)
    fig, axes = plt.subplots(2, 1, figsize=(8, 16))
    axes = axes.flatten()
    ax1 = axes[0]
    ax2 = axes[1]
    # noinspection PyTypeChecker
    labels = np.argwhere(usps['data_labels'].T == 1)[:, 1]
    ax1.scatter(*proj[:-20].T, c=labels, cmap=plt.cm.Set1)
    ax1.scatter(*proj[-20:-10].T, c='k', marker='+', s=100,
                label='KM centroids')
    ax1.legend()

    cax = ax2.scatter(*proj[:-20].T, c=labels, cmap=plt.cm.Set1)
    ax2.scatter(*proj[-10:].T, c='k', marker='^', s=100, label='GMM centroids')
    ax1.legend()
    ticks = sorted(np.unique(labels))
    cbar = fig.colorbar(cax, ticks=ticks, orientation='horizontal')
    cbar.ax.set_yticklabels(ticks)
    plt.tight_layout()
    plt.savefig('assignment9_2.png')
    plt.show()

    y_pred = km_cluster_idx(usps, r)
    print(classification_report(labels, y_pred))

    y_pred = gmm_cluster_idx(X, gmm_pi, gmm_mu, gmm_sigma)
    print(classification_report(labels, y_pred))


############################### Assignment 9 ##################################


############################### Assignment 10 #################################
# noinspection PyPep8Naming
def gammaidx(X, k):
    dists = squareform(pdist(X))
    dists = np.partition(dists, kth=k, axis=1)[:, :k+1]
    dists = np.sort(dists, axis=1)[:, 1:]
    y = dists.mean(axis=1)
    return y


def auc(y_true, y_val, plot=False, fname: str=None):
    sorted_idx = np.argsort(y_val)[::-1]
    y_val = y_val[sorted_idx]
    y_true: np.ndarray = y_true[sorted_idx] == 1
    thresholds = np.copy(y_val)
    br_thresholds = np.broadcast_to(thresholds, (len(y_val), len(y_val))).T
    y_pred = y_val >= br_thresholds
    tp = (y_pred & y_true).sum(axis=1)
    fp = (y_pred & ~y_true).sum(axis=1)

    tpr: np.ndarray = tp / len(y_true[y_true])
    fpr: np.ndarray = fp / len(y_true[~y_true])

    if plot:
        plt.figure(figsize=(8, 8))
        plt.scatter(fpr, tpr, s=100, c='r')
        plt.plot(*np.r_['0,2', 0:1.1:.1, 0:1.1:.1], '--', alpha=0.7)
        plt.ylim(0, 1)
        plt.ylabel("True positive rate")
        plt.xlim(0, 1)
        plt.xlabel("False positive rate")
        if fname is None:
            fname = 'auc.png'
        plt.savefig(fname)
        plt.show()

    c = np.trapz(tpr, fpr)
    return c


# noinspection PyPep8Naming
def in_score(X, pi, mu, sigma, tol=1e-5):
    N, _ = X.shape
    k, _ = mu.shape
    gamma = np.zeros((k, N))
    likelihood = np.zeros(N)

    for centr in range(k):
        g = norm_pdf(X, mu[centr], sigma[centr], tol=tol)
        gamma[centr] = pi[centr] * g
        likelihood += pi[centr] * g ** 2
    likelihood /= gamma.sum(axis=0)
    return np.log(likelihood)


def assignment10():
    lab_data = np.load('./lab_data.npz')
    gmm_pi, gmm_mu, gmm_sigma, _ = em_gmm(lab_data['X'], 3, init_kmeans=True,
                                          print_progress=False, tol=1e-1)
    print(gmm_mu)

    scores: np.ndarray = np.zeros(len(lab_data['X']) - 1)
    sk_scores: np.ndarray = np.zeros(len(lab_data['X']) - 1)
    for k in range(1, len(lab_data['X'])):
        y_pred = gammaidx(lab_data['X'], k)
        scores[k - 1] = auc(lab_data['Y'], y_pred)
        sk_scores[k - 1] = roc_auc_score(lab_data['Y'], y_pred)

    k = scores.argmax() + 1
    print(f"Own max AUC score: {scores.max()}; k: {k}")
    print(f"SKLearn max AUC score: {sk_scores.max()}; "
          f"k: {sk_scores.argmax() + 1}")
    y_pred = -gammaidx(lab_data['X'], k)
    auc(lab_data['Y'], y_pred, True, fname='assignment10_3.png')

    y_pred = in_score(lab_data['X'], gmm_pi, gmm_mu, gmm_sigma)
    auc_score = auc(lab_data['Y'], y_pred, True, fname='assignment10_4.png')
    print(f"k: {3}. AUC: {auc_score}")

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(*lab_data['X'][lab_data['Y'] == 1].T, label='Inliers')
    ax.scatter(*lab_data['X'][lab_data['Y'] == -1].T, label='Outliers')
    ax.legend()
    plt.savefig('assignment10.png')
    plt.show()
############################### Assignment 10 #################################


if __name__ == '__main__':
    assignment7()
    assignment8()
    assignment9()
    assignment10()
