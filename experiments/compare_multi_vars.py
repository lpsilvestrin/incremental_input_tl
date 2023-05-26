import numpy as np
import pandas as pd
import seaborn as sns
import re
from matplotlib import pyplot as plt
from scipy.linalg import lstsq
from scipy.stats import ttest_rel, shapiro, wilcoxon

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


def airfare():
    """
    sample disjoint source, target and test datasets for each fold.
    use "time" as dependent variable and meal.cal as missing variable

    :param ns:
    :param nt: size of target set
    :param ntst: size of test set
    :param nf: nb of folds
    :return:
    """
    air = pd.read_csv("~/datasets/airfare_prices/Consumer_Airfare_Report__Table_1_-_Top_1_000_Contiguous_State_City-Pair_Markets.csv",
                      usecols=["nsmiles", "large_ms", "fare"])
    ov = "nsmiles"      # old variable
    mv = "large_ms"     # missing variable
    pv = "fare"     # dependent variable
    air = air[air.notna().all(axis=1)]
    # move missing variable to the right-most column
    air = air[[ov, mv, pv]]
    return air, [mv], pv


def wine():
    wdf = pd.read_csv("~/datasets/wine_quality/winequality-red.csv", sep=';',
                      usecols=["alcohol", "volatile acidity", "quality"])
    ov = "alcohol"
    mv = "volatile acidity"     # missing variable
    pv = "quality"     # dependent variable
    wdf = wdf[wdf.notna().all(axis=1)]
    # move missing variable to the right-most column
    wdf = wdf[[ov, mv, pv]]
    return wdf


def wine4v():
    wdf = pd.read_csv("~/datasets/wine_quality/winequality-red.csv", sep=';',
                      usecols=["alcohol", "residual sugar", "sulphates", "volatile acidity", "quality"])
    ov = ["alcohol", "residual sugar"]
    mv = ["sulphates", "volatile acidity"]     # missing variable
    pv = "quality"     # dependent variable
    wdf = wdf[wdf.notna().all(axis=1)]
    # move missing variable to the right-most column
    wdf = wdf[[*ov, *mv, pv]]
    return wdf, mv, pv


def concrete():
    df = pd.read_csv("~/datasets/concrete/Concrete_Data.csv", sep=';',
                      usecols=["Cement", "Superplasticizer", "Mpa"])
    ov = "Cement"           # old variable
    mv = "Superplasticizer"  # missing variable
    pv = "Mpa"              # dependent variable
    df = df[df.notna().all(axis=1)]
    # move missing variable to the right-most column
    df = df[[ov, mv, pv]]
    return df


def concrete4v():
    df = pd.read_csv("~/datasets/concrete/Concrete_Data.csv", sep=';',
                     usecols=["Cement", "Age", "Water", "Superplasticizer", "Mpa"])
    ov = ["Cement", "Age"]  # old variable
    mv = ["Water", "Superplasticizer"]  # missing variable
    pv = "Mpa"  # dependent variable
    df = df[df.notna().all(axis=1)]
    # move missing variable to the right-most column
    df = df[[*ov, *mv, pv]]
    return df, mv, pv


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


def compute_mi(Xs, Ys, Xt, Yt):
    # compute mean-inputation method
    ds = Xs.shape[1]
    mu = np.mean(Xt, axis=0)
    xs_mi = np.concatenate([Xs, mu[ds:]*np.ones([len(Xs),Xt.shape[1]-ds])], axis=1)
    x_mi = adi(np.concatenate([xs_mi, Xt]))
    y_mi = np.concatenate([Ys, Yt])

    w_mi = alg.SolveRidgeRegression(x_mi, y_mi, 0, lam=.0)
    return w_mi

def compute_ols(Xs, Ys, Xt, Yt):
    ixt = adi(Xt)
    wt, _, _, _ = lstsq(ixt, Yt)
    return wt

def compute_gwot(Xs, Ys, Xt, Yt):
    # compute gromov-wasserstein transported source data
    xs_gwot = alg.gwot(Xs, Xt)
    x_gwot = np.concatenate([xs_gwot, Xt])
    x_gwot = adi(x_gwot)
    y_gwot = np.concatenate([Ys, Yt])

    w_gwot = alg.SolveRidgeRegression(x_gwot, y_gwot, 0, lam=.0)
    return w_gwot

def compute_egwot(Xs, Ys, Xt, Yt):
    # compute transported source data using entropic gromov-wasserstein ot
    xs_egwot = alg.egwot(Xs, Xt)
    x_egwot = np.concatenate([xs_egwot, Xt])
    x_egwot = adi(x_egwot)
    y_egwot = np.concatenate([Ys, Yt])

    w_egwot = alg.SolveRidgeRegression(x_egwot, y_egwot, 0, lam=.0)
    return w_egwot

def compute_mice_rf_lstsq(Xs, Ys, Xt, Yt):
    x_micerf, _ = alg.multiple_inputation_forest(Xs, Xt)
    y_micerf = np.concatenate([Ys, Yt])
    x_micerf = adi(x_micerf)
    w_micerf = alg.SolveRidgeRegression(x_micerf, y_micerf, 0, lam=.0)
    return w_micerf

def compute_mice_rf(Xs, Ys, Xt, Yt):
    yxs = np.concatenate([Ys.reshape(-1,1), Xs], axis=1)
    yxt = np.concatenate([Yt.reshape(-1,1), Xt], axis=1)
    _, kernel = alg.multiple_inputation_forest(yxs, yxt)
    return kernel

def compute_dsft_lstsq(Xs, Ys, Xt, Yt):
    ds = Xs.shape[1]
    xtt = Xt[:, ds:]
    xtc = Xt[:, :ds]
    p1, p2, p0 = alg.dsft_lstsq(Xs, xtc, xtt, Ys, Yt)
    return np.concatenate([p1, p2, [p0]])

def compute_dsft_lstsq_v2(Xs, Ys, Xt, Yt):
    ds = Xs.shape[1]
    xtt = Xt[:, ds:]
    xtc = Xt[:, :ds]
    p1, p2, p0 = alg.dsft_lstsq_v2(Xs, xtc, xtt, Ys, Yt)
    return np.concatenate([p1, p2, [p0]])

def compute_dsft_solver(Xs, Ys, Xt, Yt):
    ds = Xs.shape[1]
    xtt = Xt[:, ds:]
    xtc = Xt[:, :ds]
    w_dsft_solver = alg.dsft_nlp(Xs, xtc, xtt, Ys, Yt)
    return w_dsft_solver

def compute_dsft_ws_solver(Xs, Ys, Xt, Yt):
    ds = Xs.shape[1]
    xtt = Xt[:, ds:]
    xtc = Xt[:, :ds]
    w_dsft_ws_solver = alg.dsft_ws(Xs, xtc, xtt, Ys, Yt)
    return w_dsft_ws_solver

def compute_dsft_lstsq_v3(Xs, Ys, Xt, Yt):
    ds = Xs.shape[1]
    xtt = Xt[:, ds:]
    xtc = Xt[:, :ds]
    return alg.dsft_lstsq_v3(Xs, xtc, xtt, Ys, Yt)


def compute_entropy_lstsq(Xs, Ys, Xt, Yt):
    ns, ds = Xs.shape
    nt, dt = Xt.shape
    x = np.zeros([ns + nt, dt])
    x[:ns, :ds] = Xs
    x[ns:] = Xt
    y = np.concatenate([Ys, Yt])
    w = alg.entropy_lstsq(adi(x), y.reshape(ns + nt, -1), loss='MEE')
    outputs = np.dot(adi(x), w)
    residuals = np.mean(y - outputs)
    w[-1] += residuals
    return w


def compute_MI(Xs, Ys, Xt, Yt):
    ns, ds = Xs.shape
    nt, dt = Xt.shape
    x = np.zeros([ns + nt, dt])
    x[:ns, :ds] = Xs
    x[ns:] = Xt
    y = np.concatenate([Ys, Yt])
    return alg.entropy_lstsq(adi(x), y.reshape(ns+nt,-1), loss='MI')


def compute_HSIC(Xs, Ys, Xt, Yt):
    ns, ds = Xs.shape
    nt, dt = Xt.shape
    x = np.zeros([ns + nt, dt])
    x[:ns, :ds] = Xs
    x[ns:] = Xt
    y = np.concatenate([Ys, Yt])
    return alg.entropy_lstsq(adi(x), y.reshape(ns+nt,-1), loss='HSIC')


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

def mse_mice_rf(kernel, x_test, y_test):
    yx = np.concatenate([np.nan*np.ones([len(y_test),1]), x_test], axis=1)
    pred = kernel.impute_new_data(yx).complete_data(dataset=0)[:,0]
    err = mean_squared_error(y_test, pred)
    r2 = r2_score(y_test, pred)
    return err, r2

def plot_sota_comparison():
    """
    compare the MSE of different transfer learning methods in the incremental case using different
    real life datasets using a fixed source and test sets and varying the target dataset size.
    rerun with multiple independent samples of target data for each size
    :return:
    """
    Ns = 100        # nb of samples for the source dataset
    Nt = 12         # max samples for the target dataset
    min_Nt = 8      # minimal nb of samples for the target dataset
    Ntst = 700      # nb of samples for the test dataset
    n_folds = 25    # a.k.a. nb of repetitions
    df, mv, pv = wine4v()
    ptitle = "Wine"

    results = []

    # draw random datasets with specified sizes
    sizes = range(min_Nt, Nt+1)
    data_draws = [lin_sim(Ns, n, Ntst, n_folds, df, mv, pv) for n in sizes]

    for data, n in zip(data_draws, sizes):
        # params = [compute_params(*d[:4]) for d in data]
        res_dsftnl = [mse(compute_dsft_nl(*d[:4]), *d[4:]) for d in data]
        res_dsft = [mse(compute_dsft(*d[:4]), *d[4:]) for d in data]
        res_dp = [mse2(*compute_dp(*d[:4]), *d[4:]) for d in data]
        # res_mimpute = [mse(p[4], *d[4:]) for p, d in zip(params, data)]
        # res_ols = [mse(compute_ols(*d[:4]), *d[4:]) for d in data]
        # res_gwot = [mse(p, *d[4:]) for p, d in zip(compute_gwot(*d[:4]), data)]
        # res_egwot = [mse(p, *d[4:]) for p, d in zip(compute_egwot(*d[:4]), data)]
        # res_micerf_lstsq = [mse(compute_mice_rf_lstsq(*d[:4]), *d[4:]) for d in data]
        # res_micerf = [mse_mice_rf(compute_mice_rf(*d[:4]), *d[4:]) for d in data]
        # res_dsft_lstsq = [mse(compute_dsft_lstsq(*d[:4]), *d[4:]) for d in data]
        # res_dsft_lstsq2 = [mse(compute_dsft_lstsq_v2(*d[:4]), *d[4:]) for d in data]
        # res_dsft_solver = [mse(compute_dsft_solver(*d[:4]), *d[4:]) for d in data]
        # res_dsft_ws_solver = [mse(compute_dsft_ws_solver(*d[:4]), *d[4:]) for d in data]
        # res_dsft_lstsq3 = [mse(compute_dsft_lstsq_v3(*d[:4]), *d[4:]) for d in data]

        # compute parameters and gain for each fold
        results.extend([[n, "dp", r[0], r[1]] for r in res_dp])
        results.extend([[n, "dsft", r[0], r[1]] for r in res_dsft])
        results.extend([[n, "dsft_nl", r[0], r[1]] for r in res_dsftnl])
        # results.extend([[n, "ols", r[0], r[1]] for r in res_ols])
        # results.extend([[n, "min", r[0], r[1]] for r in res_mimpute])
        # results.extend([[n, "gwot", r[0], r[1]] for r in res_gwot])
        # results.extend([[n, "egwot", r[0], r[1]] for r in res_egwot])
        # results.extend([[n, "mice", r[0], r[1]] for r in res_micerf])
        # results.extend([[n, "mice+lstsq", r[0], r[1]] for r in res_micerf_lstsq])
        # results.extend([[n, "dsft_lstsq", r[0], r[1]] for r in res_dsft_lstsq])
        # results.extend([[n, "dsft_lstsq2", r[0], r[1]] for r in res_dsft_lstsq2])
        # results.extend([[n, "dsft_nl*", r[0], r[1]] for r in res_dsft_solver])
        # results.extend([[n, "dsft_ws", r[0], r[1]] for r in res_dsft_ws_solver])
        # results.extend([[n, "dsft_lstsq3", r[0], r[1]] for r in res_dsft_lstsq3])

    r2_label = "$R^2$"
    results = pd.DataFrame(results, columns=["$n_T$", "approach", "MSE", r2_label])

    # plt.figure(figsize=(6, 4))
    # sns.lineplot(x="$n_T$", y=y_axis_name, data=results[results["approach"] == "pooling"])
    # plt.tight_layout()
    # plt.show()
    #
    # sns.lineplot(x="$n_T$", y=r2_label, hue="approach", data=results)
    # plt.tight_layout()
    # plt.show()
    # sns.set_context("talk")
    sns.lineplot(x="$n_T$", y="MSE", hue="approach", data=results)
    plt.title(f"{ptitle} dataset")
    plt.tight_layout()
    # plt.savefig(f"figures/comparison_{ptitle}_{Ntst}tst_{n_folds}reps_{min_Nt}min.pdf", dpi=300)
    plt.show()


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

    # Ntst = 700      # nb of samples for the test dataset
    # n_folds = 25    # a.k.a. nb of repetitions
    # df, mv, pv = wine4v()

    # draw random datasets with specified sizes
    data = lin_sim(Ns, Nt, Ntst, n_folds, df, mv, pv)

    # for data, n in zip(data_draws, sizes):
    # params = [compute_params(*d[:4]) for d in data]
    res_ols = [mse(compute_ols(*d[:4]), *d[4:]) for d in data]
    res_dsftnl = [mse(compute_dsft_nl(*d[:4]), *d[4:]) for d in data]
    res_dsft = [mse(compute_dsft(*d[:4]), *d[4:]) for d in data]
    res_dp = [mse2(*compute_dp(*d[:4]), *d[4:]) for d in data]
    # res_entropy = [mse(compute_entropy_lstsq(*d[:4]), *d[4:]) for d in data]
    # res_MI = [mse(compute_MI(*d[:4]), *d[4:]) for d in data]
    # res_HSIC = [mse(compute_HSIC(*d[:4]), *d[4:]) for d in data]
    # res_micerf_lstsq = [mse(compute_mice_rf_lstsq(*d[:4]), *d[4:]) for d in data]

    results = pd.DataFrame()
    results['ols'] = [r[0] for r in res_ols]
    results["dsft"] = [r[0] for r in res_dsft]
    results["dsft_nl"] = [r[0] for r in res_dsftnl]
    results['dp'] = [r[0] for r in res_dp]
    # results['mice'] = [r[0] for r in res_micerf_lstsq]
    # results['entropy'] = [r[0] for r in res_entropy]
    # results['MI'] = [r[0] for r in res_MI]
    # results['HSIC'] = [r[0] for r in res_HSIC]
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

        def ttest(m1, m2):
            stat, p = ttest_rel(m1, m2)
            if p >= 0.05 / 3:
                test_result = '?'
            elif stat < 0:
                test_result = 'v'
            else:
                test_result = 'x'
            return test_result

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
    # plot_housing_data()
    # plot_airfare2()
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