import numpy as np
from sklearn.metrics.pairwise import rbf_kernel
import ot
import scipy as sp
import miceforest as mf

from scipy.optimize import minimize
from dsft_lstq_solver import DSFT_lstsq
from dsft_lstsq_solver_v2 import DSFT_v2_NLP
from dsft_solver import DSFT_solver
from dsft_sliced_wasserstein_solver import DSFT_sw_solver
from dsft_lstsq_v3_solver import DSFT_lstsq_v3
from entropy_loss_solver import Entropy_loss_solver


def SolveRidgeRegression(X, y, n_s, lam=0., alpha=1.):
    """
    code adapted from https://anujkatiyal.com/blog/2017/09/30/ml-regression/
    :param Xt:
    :param yt:
    :param Xs:
    :param ys:
    :param lam:
    :return:
    """
    def num_denom(X, y):
        xtranspose = np.transpose(X)
        denominator = np.dot(xtranspose, X)
        # if xtransx.shape[0] != xtransx.shape[1]:
        #     raise ValueError('Needs to be a square matrix for inverse')
        # lamidentity = np.identity(xtransx.shape[0]) * lam
        # numerator = xtransx + lamidentity
        numerator = np.dot(xtranspose, y)
        return numerator, denominator

    num, denom = num_denom(X[n_s:, :], y[n_s:])
    if n_s > 0:
        num_src, denom_src = num_denom(X[:n_s, :], y[:n_s])
        num = (1-alpha) * num + alpha * num_src
        denom = (1-alpha) * denom + alpha * denom_src

    lamidentity = np.identity(denom.shape[0]) * lam
    matinv = np.linalg.pinv(lamidentity + denom, hermitian=True)
    wRR = np.dot(matinv, num)
    # _, S, _ = np.linalg.svd(X)
    # df = np.sum(np.square(S) / (np.square(S) + lam))
    # wRR_list.append(wRR)
    # df_list.append(df)
    return wRR


def solve_wls(X, y, weights):
    """
    solve weighted least-squares given the data X, y and the weight matrix
    :param X: m x n matrix
    :param y: m-dimensional vector
    :param weights: m-dim vector
    :return:
    """
    def num_denom(X, y):
        xtransposeW = np.dot(np.transpose(X), weights)
        denominator = np.dot(xtransposeW, X)

        numerator = np.dot(xtransposeW, y)
        return numerator, denominator

    num, denom = num_denom(X, y)
    matinv = np.linalg.pinv(denom, hermitian=True)
    wRR = np.dot(matinv, num)

    return wRR


# implementation of the Domain Specific Feature Transfer (DSFT) method according to the paper:
# "A General Domain Specific Feature Transfer Framework for Hybrid Domain Adaptation"
def dsft(xs, xtc, xt, a=1e5, b=1):
    """
    implementation of the Domain Specific Feature Transfer (DSFT) method according to the paper:
    "A General Domain Specific Feature Transfer Framework for Hybrid Domain Adaptation"
    default hyperparameters are a=1e5 and b=1 as reported in the paper
    :param xs: source observations
    :param xtc: target observations present in the source data
    :param xt: new observations
    :param a: parameter for MMD between source and target common domain mappings
    :param b: parameter for weight normalization (l^2 norm)
    :return: return the parameter matrix Wt of the mapping m(xs) = xs: common features -> new features
    """
    ns = xs.shape[0]
    ds = xs.shape[1]
    nt = xt.shape[0]
    # transpose input data to go from (samples x features) to (features x samples)
    xt, xtc, xs = xt.T, xtc.T, xs.T
    M21 = -1/(ns*nt) * np.ones([nt, ns])
    M11 = 1/(ns*ns) * np.ones([ns, ns])
    C = np.matmul(xt, xtc.T)
    C = C - a * np.matmul(np.matmul(xt, M21), xs.T)
    D = np.matmul(xtc, xtc.T)
    D = D + a * np.matmul(np.matmul(xs, M11), xs.T)
    D = D + b * np.identity(ds)
    D = np.linalg.inv(D)
    return np.matmul(C, D).T


def dsft_rbf(xs, xtc, xt, a=1e5, b=1):
    """
    implementation of the non-linear version of the Domain Specific Feature Transfer (DSFT) method
    according to the paper:
    "A General Domain Specific Feature Transfer Framework for Hybrid Domain Adaptation"
    default hyperparameters are a=1e5 and b=1 as reported in the paper
    it uses the RBF kernel, which performed better according to the paper
    :param xs: source observations
    :param xtc: target observations present in the source data
    :param xt: new observations
    :param a: parameter for MMD between source and target common domain mappings
    :param b: parameter for weight normalization (l^2 norm)
    :return: return the values to input into the missing features in the source dataset
    """
    ns = xs.shape[0]
    nt = xt.shape[0]
    ktt = rbf_kernel(xtc, xtc)
    kts = rbf_kernel(xtc, xs)
    # transpose input data to go from (samples x features) to (features x samples)
    xt, xtc, xs = xt.T, xtc.T, xs.T
    M21 = -1/(ns*nt) * np.ones([nt, ns])
    M11 = 1/(ns*ns) * np.ones([ns, ns])
    C = np.matmul(xt, ktt)
    C = C - a * np.matmul(np.matmul(xt, M21), kts.T)
    D = np.matmul(ktt, ktt)
    D = D + a * np.matmul(np.matmul(kts, M11), kts.T)
    D = D + b * np.identity(nt)
    D = np.linalg.inv(D)
    q = np.matmul(C, D).T
    return np.matmul(kts.T, q)


def gwot(xs, xt):
    C1 = sp.spatial.distance.cdist(xs, xs)
    C2 = sp.spatial.distance.cdist(xt, xt)
    C1 /= C1.max()
    C2 /= C2.max()

    p = ot.unif(len(xs))
    q = ot.unif(len(xt))

    gw = ot.gromov.gromov_wasserstein(C1, C2, p, q, 'square_loss', verbose=False, log=False)

    return len(xs) * np.matmul(gw, xt)


def egwot(xs, xt):
    C1 = sp.spatial.distance.cdist(xs, xs)
    C2 = sp.spatial.distance.cdist(xt, xt)
    C1 /= C1.max()
    C2 /= C2.max()

    p = ot.unif(len(xs))
    q = ot.unif(len(xt))

    gw = ot.gromov.entropic_gromov_wasserstein(C1, C2, p, q, 'square_loss', epsilon=1e-3, verbose=False, log=False)

    return len(xs) * np.matmul(gw, xt)


def multiple_inputation_forest(xs, xt):
    xxs = np.concatenate([xs, np.nan*np.ones([len(xs), xt.shape[1]-xs.shape[1]])], axis=1)
    x = np.concatenate([xxs, xt])
    kds = mf.ImputationKernel(x, datasets=10, random_state=42, train_nonmissing=True, mean_match_candidates=5)
    # tune hyperparameters
    optimal_params, loss = kds.tune_parameters(dataset=0, optimization_steps=5)
    kds.mice(5, variable_parameters=optimal_params)

    return kds.complete_data(dataset=0), kds


def dsft_lstsq(xs, xtc, xtt, ys, yt, a=1e4, b=1):
    """
    original DSFT+kernel objective combined with least-squares on source and target datasets
    computed using an optimization solver
    :param xs:
    :param xtc:
    :param xtt:
    :param ys:
    :param yt:
    :param a:
    :param b:
    :return:
    """
    ns = xs.shape[0]
    nt = xtc.shape[0]
    ktt = rbf_kernel(xtc, xtc)
    kts = rbf_kernel(xtc, xs)
    # transpose input data to go from (samples x features) to (features x samples)
    # xt, xtc, xs = xt.T, xtc.T, xs.T
    M12 = -1 / (ns * nt) * np.ones([ns, nt])
    M22 = 1 / (nt*nt) * np.ones([nt, nt])
    M11 = 1 / (ns * ns) * np.ones([ns, ns])

    # add intercept at the end of the common feature matrices
    xs_ = np.pad(xs, [[0,0], [0,1]], constant_values=1)
    xtc_ =  np.pad(xtc, [[0,0], [0,1]], constant_values=1)

    # Kts, Ktt, S, Tt, Tc, M11, M22, M12, ys, yt, beta, alpha, np
    NLP = DSFT_lstsq(kts, ktt, xs_, xtt, xtc_, M11, M22, M12, ys, yt, b, a, np)
    x0 = NLP.getStartingPoint()
    lb = NLP.getLowerBounds()
    ub = NLP.getUpperBounds()
    # result = minimize(NLP.fAndG, x0, jac=True, method='SLSQP', bounds=list(zip(lb, ub)))
    result = minimize(NLP.f, x0, jac=False, method='SLSQP', bounds=list(zip(lb, ub)))

    # assemble solution and map back to original problem
    Q, p1, p2 = NLP.variables(result.x)
    # separate intercept from the other parameters
    p0 = p1[-1]
    p1 = p1[:-1]
    return p1, p2, p0


def dsft_lstsq_v2(xs, xtc, xtt, ys, yt, a=1e5, b=1, g=1e-2):
    """
    original DSFT+kernel objective combined with least-squares on source and target datasets
    computed using an optimization solver
    removed the part of the original DSFT loss minimizing the squared error on the new observed features
    :param xs:
    :param xtc:
    :param xtt:
    :param ys:
    :param yt:
    :param a:
    :param b:
    :return:
    """

    ns = xs.shape[0]
    nt = xtc.shape[0]
    ktt = rbf_kernel(xtc, xtc)
    kts = rbf_kernel(xtc, xs)
    # transpose input data to go from (samples x features) to (features x samples)
    # xt, xtc, xs = xt.T, xtc.T, xs.T
    M12 = -1 / (ns * nt) * np.ones([ns, nt])
    M22 = 1 / (nt * nt) * np.ones([nt, nt])
    M11 = 1 / (ns * ns) * np.ones([ns, ns])

    # add intercept at the end of the common feature matrices
    xs_ = np.pad(xs, [[0, 0], [0, 1]], constant_values=1)
    xtc_ = np.pad(xtc, [[0, 0], [0, 1]], constant_values=1)

    # Kts, Ktt, S, Tt, Tc, M11, M22, M12, ys, yt, beta, alpha, gamma, np
    NLP = DSFT_v2_NLP(kts, ktt, xs_, xtt, xtc_, M11, M22, M12, ys, yt, b, a, g, np)
    x0 = NLP.getStartingPoint()
    lb = NLP.getLowerBounds()
    ub = NLP.getUpperBounds()
    result = minimize(NLP.fAndG, x0, jac=True, method='SLSQP',
                      bounds=list(zip(lb, ub)))

    # assemble solution and map back to original problem
    Q, p1, p2 = NLP.variables(result.x)
    # separate intercept from the other parameters
    p0 = p1[-1]
    p1 = p1[:-1]
    return p1, p2, p0


def dsft_nlp(xs, xtc, xtt, ys, yt, a=1e5, b=1):
    """
    vanilla DSFT+kernel objective computed using an NLP solver
    the optimization result is used to impute the source data and then compute the least-squares estimator
    using the combined dataset
    :param xs:
    :param xtc:
    :param xtt:
    :param ys:
    :param yt:
    :param a:
    :param b:
    :return: least-squares regression parameters
    """
    ns = xs.shape[0]
    nt = xtc.shape[0]
    ktt = rbf_kernel(xtc, xtc)
    kts = rbf_kernel(xtc, xs)
    # transpose input data to go from (samples x features) to (features x samples)
    # xt, xtc, xs = xt.T, xtc.T, xs.T
    M12 = -1 / (ns * nt) * np.ones([ns, nt])
    M22 = 1 / (nt*nt) * np.ones([nt, nt])
    M11 = 1 / (ns * ns) * np.ones([ns, ns])

    # add intercept at the end of the common feature matrices
    xtc_ = np.pad(xtc, [[0,0], [0,1]], constant_values=1)

    # Kts, Ktt, Tt, Tc, M11, M22, M12, beta, alpha, np
    NLP = DSFT_solver(kts, ktt, xtt, xtc_, M11, M22, M12, b, a, np)
    x0 = NLP.getStartingPoint()
    lb = NLP.getLowerBounds()
    ub = NLP.getUpperBounds()
    result = minimize(NLP.fAndG, x0, jac=True, method='SLSQP',
                      bounds=list(zip(lb, ub)))

    # assemble solution and map back to original problem
    Q = NLP.variables(result.x)
    # compute values to impute in the missing
    imputs = np.matmul(kts.T, Q)

    # create new homogeneous training set
    xs = np.concatenate([xs, imputs], axis=1)
    xt = np.concatenate([xtc, xtt], axis=1)
    X = np.concatenate([xs, xt])
    X = np.pad(X, [[0,0], [0,1]], constant_values=1)
    Y = np.concatenate([ys, yt])

    w = SolveRidgeRegression(X, Y, 0, lam=.0)

    return w


def dsft_ws(xs, xtc, xtt, ys, yt, a=1e5, b=0):
    """
    DSFT+kernel using sliced wasserstein distance instead of MMD
    computed using an NLP solver
    the optimization result is used to impute the source data and then compute the least-squares estimator
    using the combined dataset
    :param xs:
    :param xtc:
    :param xtt:
    :param ys:
    :param yt:
    :param a:
    :param b:
    :return: least-squares regression parameters
    """
    ns = xs.shape[0]
    nt = xtc.shape[0]
    ktt = rbf_kernel(xtc, xtc)
    kts = rbf_kernel(xtc, xs)
    # transpose input data to go from (samples x features) to (features x samples)
    # xt, xtc, xs = xt.T, xtc.T, xs.T
    M12 = -1 / (ns * nt) * np.ones([ns, nt])
    M22 = 1 / (nt*nt) * np.ones([nt, nt])
    M11 = 1 / (ns * ns) * np.ones([ns, ns])

    # add intercept at the end of the common feature matrices
    xtc_ = np.pad(xtc, [[0,0], [0,1]], constant_values=1)

    # Kts, Ktt, Tt, Tc, M11, M22, M12, beta, alpha, np
    NLP = DSFT_sw_solver(kts, ktt, xtt, xtc_, M11, M22, M12, b, a, np)
    x0 = NLP.getStartingPoint()
    lb = NLP.getLowerBounds()
    ub = NLP.getUpperBounds()
    result = minimize(NLP.f, x0, jac=False, method='SLSQP',
                      bounds=list(zip(lb, ub)))

    # assemble solution and map back to original problem
    Q = NLP.variables(result.x)
    # compute values to impute in the missing
    imputs = np.matmul(kts.T, Q)

    # create new homogeneous training set
    xs = np.concatenate([xs, imputs], axis=1)
    xt = np.concatenate([xtc, xtt], axis=1)
    X = np.concatenate([xs, xt])
    X = np.pad(X, [[0,0], [0,1]], constant_values=1)
    Y = np.concatenate([ys, yt])

    w = SolveRidgeRegression(X, Y, 0, lam=.0)

    return w


def dsft_lstsq_v3(xs, xtc, xtt, ys, yt, a=1e2):
    """
    least-squares with data filled up using only the MMD distance between imputed values and target observations
    :param xs:
    :param xtc:
    :param xtt:
    :param ys:
    :param yt:
    :param a:
    :return: least-squares regression parameters
    """
    ns = xs.shape[0]
    nt = xtc.shape[0]
    kts = rbf_kernel(xtc, xs)
    # transpose input data to go from (samples x features) to (features x samples)
    # xt, xtc, xs = xt.T, xtc.T, xs.T
    M12 = -1 / (ns * nt) * np.ones([ns, nt])
    M11 = 1 / (ns * ns) * np.ones([ns, ns])

    # add intercept at the beginning of the common feature matrices
    xtc_ = np.pad(xtc, [[0,0], [1,0]], constant_values=1)
    xs_ = np.pad(xs, [[0,0], [1,0]], constant_values=1)

    # Kts, S, Tt, Tc, M11, M12, ys, yt, alpha, np
    NLP = DSFT_lstsq_v3(kts, xs_, xtt, xtc_, M11, M12, ys, yt, a, np)
    # NLP = DSFT_lstsq_v3(xs.T, xs_, xtt, xtc_, M11, M12, ys, yt, a, np)
    _, w = NLP.solve()

    # shift to place intercept in the right-most position
    w = sp.ndimage.shift(w, [-1], mode="grid-wrap")

    return w


def entropy_lstsq(x, y, loss='MEE'):
    _, w = Entropy_loss_solver(x, y, loss, np).solve()
    return w


if __name__ == '__main__':
    def test1():
        xs = np.random.randn(10, 4)
        xtc = np.random.randn(5, 4)
        xt = np.random.randn(5, 2)
        wt = dsft_rbf(xs, xtc, xt)
        print(wt)


    def test2():
        xs = np.random.randn(10, 4)
        xt = np.random.randn(5, 6)
        xs_trans, kernel = multiple_inputation_forest(xs, xt)
        xs_miss = np.concatenate([xs, np.nan * np.ones([10, 2])], axis=1)
        xs_inpute = kernel.impute_new_data(xs_miss)

        print(xs_inpute)

    test2()
