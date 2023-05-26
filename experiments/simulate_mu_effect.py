import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from sklearn.metrics import mean_squared_error

from algorithms import SolveRidgeRegression, solve_wls

_SEED = 10
np.random.seed(_SEED)

sns.set_style("whitegrid")


def lin_sim(n, d, params=None, noise=0., mean=None, constant=0):
    if params is None:
        params = np.repeat(0.5, d)
    if mean is None:
        mean = np.zeros(d)
    cov = np.identity(d)
    x = np.random.multivariate_normal(mean, cov=cov, size=n)

    y = np.matmul(params.T, x.T) + noise * np.random.randn(n) + constant

    return x, y


def adi(x):
    # adds intercept to x matrix
    return np.pad(x, [[0,0], [0,1]], constant_values=1)


def plot_alpha_estim_comparison():
    """
    compare how the gain of data-pooling changes by using the real alpha vs the estimated alpha values
    :return:
    """
    theta = np.array([2., -2.])
    noise_var = 1
    constant = 2
    dmean = np.array([0, 0])

    Ns = 100
    Nt = 30

    def sim_data(n_s, n_t, p):
        x_s, y_s = lin_sim(n_s, len(p), params=p, noise=noise_var, mean=dmean, constant=constant)
        x_t, y_t = lin_sim(n_t, len(p), params=p, noise=noise_var, mean=dmean, constant=constant)
        x_s = x_s[:, :-1]
        return x_s, y_s, x_t, y_t

    _, _, x_test, y_test = sim_data(0, 1000, theta)
    x_test = adi(x_test)

    def compute_params(Xs, Ys, Xt, Yt):
        n_s = len(Xs)
        n_t = len(Xt)

        x_dp = np.zeros([n_s + n_t, 2])
        x_dp[:n_s, :-1] = Xs
        x_dp[n_s:] = Xt
        x_dp = adi(x_dp)
        y_dp = np.concatenate([Ys, Yt])

        ixt = adi(Xt)
        ixs = adi(Xs)
        wt = SolveRidgeRegression(ixt, Yt, 0, lam=.0)
        var_t = mean_squared_error(Yt, np.dot(ixt, wt))
        ws = SolveRidgeRegression(ixs, Ys, 0, lam=.0)
        var_s = mean_squared_error(Ys, np.dot(ixs, ws))

        # compute WLS estimator
        weight_vec = np.concatenate([np.repeat(1 / var_s, n_s), np.repeat(1 / var_t, n_t)])
        weights = weight_vec * np.identity(n_s + n_t)
        wWLS = solve_wls(x_dp, y_dp, weights)
        # compute true alphas
        mw = theta[1]
        ta = np.concatenate([np.repeat(1 / (noise_var + mw * mw), n_s), np.repeat(1 / noise_var, n_t)])
        wWLS2 = solve_wls(x_dp, y_dp, ta * np.identity(n_s + n_t))

        return wt, wWLS, wWLS2

    def mse(p):
        return mean_squared_error(y_test, np.dot(x_test, p))

    tmp = []
    for _ in range(200):
        # Xs, Ys, Xt, Yt = sim_data(Ns, Nt, theta)
        tmp.extend([[compute_params(*sim_data(Ns, n, theta)), n] for n in range(5, Nt+1)])
    params = [[n, "estim. $\\alpha$", mse(p[0]) - mse(p[1])] for [p, n] in tmp]
    params.extend([[n, "real $\\alpha$", mse(p[0]) - mse(p[2])] for [p, n] in tmp])
    # params.extend([[n, f"fine-tuning k = {k}", *p[2], mse(p[2])] for [p, n] in tmp])
    y_axis_name = "transfer gain"
    params = pd.DataFrame(params, columns=["$n_T$", "approach", y_axis_name])
    plt.figure(figsize=(6,4))
    sns.lineplot(x="$n_T$", y=y_axis_name, hue="approach", data=params)
    plt.tight_layout()
    plt.legend(title=None)
    plt.show()


def plot_mean_estim_comparison():
    """
    compare how the gain of data-pooling changes when E[X"] != 0 and x_t - mu has to be used instead
    :return:
    """
    theta = np.array([2., -2])
    noise_var = 1
    constant = 2
    dmean = np.array([0, 1])
    d = len(theta)
    Ns = 100
    Nt = 30

    def sim_data(n_s, n_t, p):
        x_s, y_s = lin_sim(n_s, len(p), params=p, noise=noise_var, mean=dmean, constant=constant)
        x_t, y_t = lin_sim(n_t, len(p), params=p, noise=noise_var, mean=dmean, constant=constant)
        x_s = x_s[:, :-1]
        return x_s, y_s, x_t, y_t

    _, _, x_test, y_test = sim_data(0, 1000, theta)
    x_test = adi(x_test)

    def compute_params(Xs, Ys, Xt, Yt):
        n_s = len(Xs)
        n_t = len(Xt)

        x_dp = np.zeros([n_s + n_t, d])
        x_dp[:n_s, :-1] = Xs
        mu = np.mean(Xt, axis=0)
        mu[0] = 0
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
        weight_vec = np.concatenate([np.repeat(1 / var_s, n_s), np.repeat(1 / var_t, n_t)])
        weights = weight_vec * np.identity(n_s + n_t)
        wWLS = solve_wls(x_dp, y_dp, weights)

        # compute WLS with true mean
        x_dp[n_s:, :-1] = Xt - dmean
        ixt = adi(Xt - dmean)
        wt = SolveRidgeRegression(ixt, Yt, 0, lam=.0)
        var_t = mean_squared_error(Yt, np.dot(ixt, wt))

        weights = np.diag([1/var_s]*n_s + [1/var_t]*n_t)
        wWLS2 = solve_wls(x_dp, y_dp, weights)

        # compute basic OLS estimator
        ixt = adi(Xt)
        wt = SolveRidgeRegression(ixt, Yt, 0, lam=.0)

        return wt, wWLS, wWLS2, mu

    def mse(p):
        return mean_squared_error(y_test, np.dot(x_test, p))

    def mse2(p, mu):
        _x = x_test - np.pad(mu, [0,1])
        return mean_squared_error(y_test, np.dot(_x, p))

    tmp = []
    params = []
    for _ in range(200):
        # Xs, Ys, Xt, Yt = sim_data(Ns, Nt, theta)
        tmp.extend([[compute_params(*sim_data(Ns, n, theta)), n] for n in range(5, Nt+1)])
    params.extend([[n, "estim. mean", mse(p[0]) - mse2(p[1], p[3])] for [p, n] in tmp])
    params.extend([[n, "true mean", mse(p[0]) - mse2(p[2], dmean)] for [p, n] in tmp])
    # params.extend([[n, f"fine-tuning k = {k}", *p[2], mse(p[2])] for [p, n] in tmp])
    y_axis_name = "transfer gain"
    params = pd.DataFrame(params, columns=["$n_T$", "approach", y_axis_name])
    plt.figure(figsize=(6,4))
    sns.lineplot(x="$n_T$", y=y_axis_name, hue="approach", data=params)
    plt.tight_layout()
    plt.legend(title=None)
    plt.show()


if __name__ == '__main__':
    plot_alpha_estim_comparison()
    plot_mean_estim_comparison()