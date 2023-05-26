import argparse
import re

import numpy as np
import pandas as pd
from scipy.linalg import lstsq
from scipy.stats import wilcoxon
from sklearn.model_selection import ShuffleSplit, train_test_split
from sklearn.metrics import mean_squared_error
from tqdm import tqdm
from uci_datasets import Dataset

from simulations.compare_multi_vars import drop_constant_cols
import algorithms as alg


def lin_sim(nt, ntst, nf, df, mv, pv):
    """
    sample disjoint source, target and test datasets for each fold.
    IMPORTANT: pay attention to the column order in :param df
    :param ns:
    :param nt: size of target set
    :param ntst: size of test set
    :param nf: nb of folds
    :param df: IMPORTANT: the columns should be ordered as [src var, new var, dependent var]
    :param mv: list or string containing the names of missing columns
    :param pv: string with the name of the column to predict
    :return:
    """
    ss = ShuffleSplit(n_splits=nf, test_size=ntst, random_state=1)
    folds = []

    for train_index, test_index in ss.split(df):
        x_test = df.iloc[test_index].reset_index(drop=True)
        y_test = x_test[pv].to_numpy()
        x_test = x_test.drop(columns=pv).to_numpy()
        src_index, tar_index = train_test_split(train_index, test_size=nt, shuffle=False)
        x_src = df.iloc[src_index].reset_index(drop=True)
        y_src = x_src[pv].to_numpy()
        # remove new features from source set
        x_src = x_src.drop(columns=np.concatenate([mv, [pv]])).to_numpy()
        x_tar = df.iloc[tar_index].reset_index(drop=True)
        y_tar = x_tar[pv].to_numpy()
        x_tar = x_tar.drop(columns=pv).to_numpy()
        folds.append((x_src, y_src, x_tar, y_tar, x_test, y_test))

    return folds


def compute_dp(Xs, Ys, Xt, Yt):
    n_s = len(Xs)
    n_t = len(Xt)
    d = Xt.shape[1]
    ds = Xs.shape[1]

    x_dp = np.zeros([n_s + n_t, d])
    x_dp[:n_s, :ds] = Xs
    mu = np.mean(Xt, axis=0)
    mu[:ds] = 0.
    x_dp[n_s:] = Xt - mu
    x_dp = adi(x_dp)
    y_dp = np.concatenate([Ys, Yt])

    ixt = adi(Xt - mu)
    ixs = adi(Xs)
    # wt = SolveRidgeRegression(ixt, Yt, 0, lam=.0)
    wt, var_t, _, _ = lstsq(ixt, Yt)
    var_t = mean_squared_error(Yt, np.dot(ixt, wt))
    # ws = SolveRidgeRegression(ixs, Ys, 0, lam=.0)
    ws, var_s, _, _ = lstsq(ixs, Ys)
    # var_s = var_s / n_s
    var_s = mean_squared_error(Ys, np.dot(ixs, ws))

    # compute WLS estimator
    weights = np.diag([1/var_s]*n_s + [1/var_t]*n_t)
    wWLS = alg.solve_wls(x_dp, y_dp, weights)
    return wWLS, mu


def compute_dsft(Xs, Ys, Xt, Yt):
    # compute feature mapping using DSFT
    ds = Xs.shape[1]
    # compute src features, except for the non-additive feature
    fmap = alg.dsft(adi(Xs), adi(Xt[:, :ds]), Xt[:, ds:-1])
    a = np.matmul(adi(Xs), fmap)
    # compute non-additive feature for src dataset
    nas = (Xs[:, -1] * a[:, -1]).reshape(-1, 1)
    # create new homogeneous training set
    x_dsft = np.concatenate([Xs, a, nas], axis=1)
    x_dsft = np.concatenate([x_dsft, Xt])
    x_dsft = adi(x_dsft)
    y_dsft = np.concatenate([Ys, Yt])

    w_dsft = alg.SolveRidgeRegression(x_dsft, y_dsft, 0, lam=.0)

    return w_dsft


def compute_dsft_nl(Xs, Ys, Xt, Yt):
    # compute feature mapping using DSFT_nl
    ds = Xs.shape[1]
    # compute src features, except for the non-additive feature
    a = alg.dsft_rbf(adi(Xs), adi(Xt[:, :ds]), Xt[:, ds:-1])
    # compute non-additive feature for src dataset
    nas = (Xs[:, -1] * a[:, -1]).reshape(-1, 1)
    # create new homogeneous training set
    x_dsft_nl = np.concatenate([Xs, a, nas], axis=1)
    x_dsft_nl = np.concatenate([x_dsft_nl, Xt])
    x_dsft_nl = adi(x_dsft_nl)
    y_dsft_nl = np.concatenate([Ys, Yt])

    w_dsft_nl = alg.SolveRidgeRegression(x_dsft_nl, y_dsft_nl, 0, lam=.0)

    return w_dsft_nl


def compute_ols(Xs, Ys, Xt, Yt):
    ixt = adi(Xt)
    wt, _, _, _ = lstsq(ixt, Yt)
    return wt


def adi(x):
    # adds intercept to x matrix
    return np.pad(x, [[0,0], [0,1]], constant_values=1)


def mse(p, x_test, y_test):
    _x = adi(x_test)
    pred = np.dot(_x, p)
    err = mean_squared_error(y_test, pred)
    return err

def mse2(p, mu, x_test, y_test):
    _x = adi(x_test) - np.pad(mu, [0, 1])
    pred = np.dot(_x, p)
    err = mean_squared_error(y_test, pred)
    return err


def sota_comparison_non_additive(nb_missing=1, dataset_name='energy', Nt=30):
    # load the dataset and sort columns by their pearson correlation
    dataset = Dataset(dataset_name)
    x_clean = drop_constant_cols(dataset.x)
    sorted_idx = np.argsort(np.abs(np.corrcoef(x_clean, y=dataset.y, rowvar=False)[-1, :-1]))
    x_sorted = dataset.x[:, sorted_idx]

    # create feature to break non-additivity assumption
    x_nonadd = (x_sorted[:, -1] * x_sorted[:, -(nb_missing+1)]).reshape(-1, 1)
    y = dataset.y + x_nonadd
    x_sorted = np.concatenate([x_sorted, x_nonadd], axis=1)
    nb_missing = nb_missing + 1
    nb_samples, nb_features = x_sorted.shape

    df = pd.DataFrame(np.concatenate([x_sorted, y], axis=1))
    mv = np.arange(nb_features-nb_missing, nb_features)
    pv = nb_features
    Ntst = nb_samples // 10
    n_folds = 30

    # draw random datasets with specified sizes
    data = lin_sim(Nt, Ntst, n_folds, df, mv, pv)
    print("Computing OLS")
    res_ols = [mse(compute_ols(xs, ys, xt, yt), xtst, ytst) for xs, ys, xt, yt, xtst, ytst in tqdm(data)]
    print("Computing non-linear DSFT")
    res_dsftnl = [mse(compute_dsft_nl(xs, ys, xt, yt), xtst, ytst) for xs, ys, xt, yt, xtst, ytst in tqdm(data)]
    print("Computing DSFT")
    res_dsft = [mse(compute_dsft(xs, ys, xt, yt), xtst, ytst) for xs, ys, xt, yt, xtst, ytst in tqdm(data)]
    print("Computing DP")
    res_dp = [mse2(*compute_dp(xs, ys, xt, yt), xtst, ytst) for xs, ys, xt, yt, xtst, ytst in tqdm(data)]

    results = pd.DataFrame()
    results['ols'] = res_ols
    results["dsft"] = res_dsft
    results["dsft_nl"] = res_dsftnl
    results['dp'] = res_dp

    results['dataset'] = dataset_name

    return results


def table_sota_comparison(nb_missing, Nt, datasets, stat_tests=True):
    res = pd.concat([sota_comparison_non_additive(nb_missing, dataset, Nt=Nt) for dataset in datasets])
    # compute the root mean squared error
    floats = res.select_dtypes('float64')
    res[floats.columns] = floats.apply(np.sqrt)
    grouped = res.groupby('dataset').agg(['mean', 'std'])
    latex_table = grouped.to_latex(float_format="%.3g")

    latex_table = re.sub("&([\s0-9\-.+e]+)&([\s0-9\-.+e]+)", r"&\1$\\pm$\2", latex_table)

    print(latex_table)

    if stat_tests:
        methods = ['ols', 'dsft', 'dsft_nl', 'dp']
        main = 'dp'
        methods.remove(main)

        def wilcox(m1, m2):
            stat, p = wilcoxon(m1, m2)
            mean1 = np.mean(m1)
            mean2 = np.mean(m2)
            if p >= 0.05 / 3:
                test_result = '?'
            elif mean1 < mean2:
                test_result = 'v'
            else:
                test_result = 'x'
            return test_result

        def test_all_methods(results, test):
            return [test(results[main], results[m]) for m in methods]

        tresults = [[d] + test_all_methods(res[res['dataset']==d], wilcox) for d in datasets]

        columns = ['dataset'] + [f"{main} x {m}" for m in methods]
        print(pd.DataFrame(tresults, columns=columns))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--nb_missing', type=int, default=1)
    parser.add_argument('--Nt', type=int, default=100)
    args = parser.parse_args()
    datasets = [
        "skillcraft",
        "sml",
        "pol",
        # "stock",
        "pumadyn32nm",
        "kin40k",
        "parkinsons",
        "protein",
        "energy",
        # "pendulum",
        "concrete"
    ]
    datasets = sorted(datasets)
    table_sota_comparison(nb_missing=args.nb_missing, Nt=args.Nt, datasets=datasets)
