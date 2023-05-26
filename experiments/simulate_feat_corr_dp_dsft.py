import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from sklearn.metrics import mean_squared_error

from algorithms import dsft, dsft_rbf, SolveRidgeRegression, solve_wls

_SEED = 10
# np.random.seed(_SEED)

sns.set_style("whitegrid")


def lin_sim(n, d, params=None, noise=0., mean=None, constant=0, corr=None):
    """
    sample gaussian data and the linear regression labels with additive gaussian noise
    after simulating the labels, the covariate matrix is replaced by some linear
    combination of the original covariates to simulate correlation

    :param n: sample size
    :param d: nb of inptus
    :param params: regression parameters
    :param noise: variance of the gaussian noise
    :param mean: d-dimensional vector with means of the features
    :param constant: additive bias of the labels
    :param corr: matrix w/ input correlation coefficients
    :return: features, labels
    """

    if params is None:
        params = np.repeat(0.5, d)
    if mean is None:
        mean = np.zeros(d)
    if corr is None:
        corr = np.identity(d)
    # cov = np.identity(d)
    x = np.random.multivariate_normal(mean, cov=corr, size=n)

    y = np.matmul(params.T, x.T) + noise * np.random.randn(n) + constant

    return x, y


def adi(x):
    # adds intercept to x matrix
    return np.pad(x, [[0,0], [0,1]], constant_values=1)


def plot_corr_comparison():
    """
    compare data-pooling against dsft in the case where the second input is a function of the first one
    i.e. x2 = x1*coef + w*(1-coef), where x1 and w follow a standard normal distribution
    :return:
    """
    theta = np.array([2., -2.])
    noise_var = 1
    constant = 2
    dmean = np.array([0, 1])
    repetitions = 200
    corr = np.array([[1, .9],
                     [.9, 1]])
    uncorr = np.identity(2)

    Ns = 100
    Nt = 8

    def sim_data(n_s, n_t, p, corr_mat):
        x_s, y_s = lin_sim(n_s, len(p), params=p, noise=noise_var, mean=dmean, constant=constant, corr=corr_mat)
        x_t, y_t = lin_sim(n_t, len(p), params=p, noise=noise_var, mean=dmean, constant=constant, corr=corr_mat)
        x_s = x_s[:, :-1]
        return x_s, y_s, x_t, y_t

    # _, _, x_test, y_test = sim_data(0, 1000, theta)
    # x_test = adi(x_test)

    def compute_params(Xs, Ys, Xt, Yt):
        n_s = len(Xs)
        n_t = len(Xt)
        d = Xt.shape[1]
        ds = Xs.shape[1]

        x_dp = np.zeros([n_s + n_t, d])
        x_dp[:n_s, :-1] = Xs
        mu = np.mean(Xt, axis=0)
        mu[0] = 0
        # mu = np.zeros(d)
        x_dp[n_s:] = Xt - mu
        x_dp = adi(x_dp)
        y_dp = np.concatenate([Ys, Yt])

        ixt = adi(Xt - mu)
        ixs = adi(Xs)
        wt = SolveRidgeRegression(ixt, Yt, 0, lam=.0)
        var_t = mean_squared_error(Yt, np.dot(ixt, wt))
        ws = SolveRidgeRegression(ixs, Ys, 0, lam=.0)
        var_s = mean_squared_error(Ys, np.dot(ixs, ws))

        # compute WLS estimator
        weights = np.diag([1/var_s]*n_s + [1/var_t]*n_t)
        wWLS = solve_wls(x_dp, y_dp, weights)

        # compute feature mapping using DSFT
        fmap = dsft(adi(Xs), adi(Xt[:, :ds]), Xt[:, ds:])
        a = np.matmul(adi(Xs), fmap)
        # create new homogeneous training set
        x_dsft = np.concatenate([Xs, a], axis=1)
        x_dsft = np.concatenate([x_dsft, Xt])
        x_dsft = adi(x_dsft)
        y_dsft = np.concatenate([Ys, Yt])

        w_dsft = SolveRidgeRegression(x_dsft, y_dsft, 0, lam=.0)

        # compute feature mapping using DSFT_nl
        a = dsft_rbf(adi(Xs), adi(Xt[:, :ds]), Xt[:, ds:])
        # create new homogeneous training set
        x_dsft_nl = np.concatenate([Xs, a], axis=1)
        x_dsft_nl = np.concatenate([x_dsft_nl, Xt])
        x_dsft_nl = adi(x_dsft_nl)
        y_dsft_nl = np.concatenate([Ys, Yt])

        w_dsft_nl = SolveRidgeRegression(x_dsft_nl, y_dsft_nl, 0, lam=.0)

        # compute OLS estimator
        ixt = adi(Xt)
        # wt, _, _, _ = lstsq(ixt, Yt)
        wt = SolveRidgeRegression(ixt, Yt, 0)

        return w_dsft, wWLS, mu, w_dsft_nl, wt

    def mse(p, x_test, y_test):
        return mean_squared_error(y_test, np.dot(x_test, p))

    def mse2(p, mu, x_test, y_test):
        _x = x_test - np.pad(mu, [0, 1])
        return mean_squared_error(y_test, np.dot(_x, p))

    # tmp = []
    params = []
    # linearly interpolate correlation matrices used to generate the data
    for corr_mat in np.linspace(uncorr, corr, 11):
        # fix seed at each
        np.random.seed(_SEED)
        tmp = [compute_params(*sim_data(Ns, Nt, theta, corr_mat)) for _ in range(repetitions)]
        # simulate the test dataset using the same correlation setup
        _, _, xt, yt = sim_data(0, 1000, theta, corr_mat)
        xt = adi(xt)
        coef = corr_mat[1, 0]  # the coefficient of correlation between x2 and x1 (used for the plots)
        # compute the test mse for each approach
        params.extend([[coef, "dp", mse2(p[1], p[2], xt, yt)] for p in tmp])
        params.extend([[coef, "dsft", mse(p[0], xt, yt)] for p in tmp])
        params.extend([[coef, "dsft_nl", mse(p[3], xt, yt)] for p in tmp])
        params.extend([[coef, "ols", mse(p[4], xt, yt)] for p in tmp])

    y_axis_name = "MSE"
    x_axis_name = "correlation coefficient"
    params = pd.DataFrame(params, columns=[x_axis_name, "approach", y_axis_name])
    plt.figure(figsize=(6, 4))
    sns.lineplot(x=x_axis_name, y=y_axis_name, hue="approach", data=params)
    plt.tight_layout()
    plt.legend(title=None)
    plt.savefig('figures/sim_correlation_covmat.pdf', dpi=300)
    plt.show()


if __name__ == '__main__':
    # plot_uncorr_comparison()
    plot_corr_comparison()
