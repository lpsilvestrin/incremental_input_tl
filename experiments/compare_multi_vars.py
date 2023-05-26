import numpy as np
import pandas as pd
import seaborn as sns
import re

from scipy.linalg import lstsq
from scipy.stats import wilcoxon

from sklearn.metrics import mean_squared_error, r2_score

import algorithms as alg

from uci_datasets import Dataset

_SEED = 10
np.random.seed(_SEED)

sns.set_style("whitegrid")


def lin_sim(ns, nt, ntst, nf, df, mv, pv):
    """
    sample disjoint source, target and test datasets for each fold.
    use "time" as dependent variable and meal.cal as missing variable
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
    # sample fixed source and test datasets
    rtar = df.sample(frac=1, random_state=1).reset_index(drop=True)
    tmp = rtar.drop(columns=mv).iloc[:ns]
    ys = tmp[pv].to_numpy()
    Xs = tmp.drop(columns=[pv]).to_numpy()
    tmp = rtar.drop(index=list(range(ns))).reset_index(drop=True)
    X_test = tmp.iloc[:ntst]
    y_test = X_test[pv].to_numpy()
    X_test = X_test.drop(columns=pv).to_numpy()

    # remove samples used in the test set and reshuffle the remaining data
    tmp = tmp.drop(index=list(range(ntst))).sample(frac=1).reset_index(drop=True)
    folds = []

    # sample disjoint test sets
    for _ in range(nf):
        Xt = tmp.iloc[:nt]
        yt = Xt[pv].to_numpy()
        Xt = Xt.drop(columns=pv).to_numpy()
        tmp = tmp.drop(index=list(range(nt))).reset_index(drop=True)

        folds.append([Xs, ys, Xt, yt, X_test, y_test])

    return folds


def adi(x):
    # adds intercept to x matrix
    return np.pad(x, [[0,0], [0,1]], constant_values=1)


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
    fmap = alg.dsft(adi(Xs), adi(Xt[:, :ds]), Xt[:, ds:])
    a = np.matmul(adi(Xs), fmap)
    # create new homogeneous training set
    x_dsft = np.concatenate([Xs, a], axis=1)
    x_dsft = np.concatenate([x_dsft, Xt])
    x_dsft = adi(x_dsft)
    y_dsft = np.concatenate([Ys, Yt])

    w_dsft = alg.SolveRidgeRegression(x_dsft, y_dsft, 0, lam=.0)

    return w_dsft


def compute_dsft_nl(Xs, Ys, Xt, Yt):
    # compute feature mapping using DSFT_nl
    ds = Xs.shape[1]
    a = alg.dsft_rbf(adi(Xs), adi(Xt[:, :ds]), Xt[:, ds:])
    # create new homogeneous training set
    x_dsft_nl = np.concatenate([Xs, a], axis=1)
    x_dsft_nl = np.concatenate([x_dsft_nl, Xt])
    x_dsft_nl = adi(x_dsft_nl)
    y_dsft_nl = np.concatenate([Ys, Yt])

    w_dsft_nl = alg.SolveRidgeRegression(x_dsft_nl, y_dsft_nl, 0, lam=.0)

    return w_dsft_nl


def compute_ols(Xs, Ys, Xt, Yt):
    ixt = adi(Xt)
    wt, _, _, _ = lstsq(ixt, Yt)
    return wt


def mse(p, x_test, y_test):
    _x = adi(x_test)
    pred = np.dot(_x, p)
    err = mean_squared_error(y_test, pred)
    r2 = r2_score(y_test, pred)
    return err, r2


def mse2(p, mu, x_test, y_test):
    _x = adi(x_test) - np.pad(mu, [0, 1])
    pred = np.dot(_x, p)
    err = mean_squared_error(y_test, pred)
    r2 = r2_score(y_test, pred)
    return err, r2


def drop_constant_cols(x: np.array):
    nb_samples, nb_cols = x.shape
    uniques_per_col = [len(np.unique(x[:, i])) for i in range(nb_cols)]
    # remove columns with less than 1% of unique values
    cols_to_keep = [upc > 1 for upc in uniques_per_col]
    return x[:, cols_to_keep]


def experiment_settings(datasets):
    settings = []
    for dataset_name in datasets:
        dataset = Dataset(dataset_name)
        x_clean = drop_constant_cols(dataset.x)
        sorted_idx = np.argsort(np.abs(np.corrcoef(x_clean, y=dataset.y, rowvar=False)[-1, :-1]))[::-1]
        x_sorted = dataset.x[:, sorted_idx]
        nb_samples, nb_features = x_sorted.shape

        Ns, Nt, Ntst, n_folds = compute_datset_statistics(nb_samples, nb_features)
        settings.append([dataset_name, nb_samples, Ns, Nt, Ntst, nb_features, n_folds])

    settings = pd.DataFrame(settings, columns=["dataset", "N", "Ns", "Nt", "Ntst", "d", "#reps"])
    print(settings.to_latex(index=False))


def compute_datset_statistics(nb_samples, nb_features):
    # compute the number of samples for the target dataset
    Nt = nb_features + 30           # nb of samples for the target dataset
    # Nt = nb_features * 4           # nb of samples for the target dataset
    Ns = np.max([Nt * 3, 100])      # nb of samples for the source dataset

    Ntst = nb_samples // 10
    n_folds = np.min([(nb_samples - Ns - Ntst) // Nt, 50])
    return Ns, Nt, Ntst, n_folds


def row_sota_comparison(nb_missing=1, dataset_name='energy'):
    # load the dataset and sort columns by their pearson correlation
    dataset = Dataset(dataset_name)
    x_clean = drop_constant_cols(dataset.x)
    sorted_idx = np.argsort(np.abs(np.corrcoef(x_clean, y=dataset.y, rowvar=False)[-1, :-1]))
    x_sorted = dataset.x[:, sorted_idx]
    nb_samples, nb_features = x_sorted.shape

    Ns, Nt, Ntst, n_folds = compute_datset_statistics(nb_samples, nb_features)

    df = pd.DataFrame(np.concatenate([x_sorted, dataset.y], axis=1))
    mv = np.arange(nb_features-nb_missing, nb_features)
    pv = nb_features


    # draw random datasets with specified sizes
    data = lin_sim(Ns, Nt, Ntst, n_folds, df, mv, pv)

    res_ols = [mse(compute_ols(*d[:4]), *d[4:]) for d in data]
    res_dsftnl = [mse(compute_dsft_nl(*d[:4]), *d[4:]) for d in data]
    res_dsft = [mse(compute_dsft(*d[:4]), *d[4:]) for d in data]
    res_dp = [mse2(*compute_dp(*d[:4]), *d[4:]) for d in data]

    results = pd.DataFrame()
    results['ols'] = [r[0] for r in res_ols]
    results["dsft"] = [r[0] for r in res_dsft]
    results["dsft_nl"] = [r[0] for r in res_dsftnl]
    results['dp'] = [r[0] for r in res_dp]

    results['dataset'] = dataset_name

    return results


def table_sota_comparison(nb_missing, datasets, stat_tests=True):
    res = pd.concat([row_sota_comparison(nb_missing, dataset) for dataset in datasets])
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
    # plot_sota_comparison()
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
    table_sota_comparison(nb_missing=3, datasets=datasets)
    # experiment_settings(datasets)