import argparse
import re

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.model_selection import ShuffleSplit, train_test_split
from tqdm import tqdm
from uci_datasets import Dataset

from experiments.compare_multi_vars import drop_constant_cols, mse, compute_ols, compute_dsft_nl, compute_dsft, \
    compute_dp, mse2


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


def sota_comparison_no_subsample(nb_missing=1, dataset_name='energy', Nt=30):
    # load the dataset and sort columns by their pearson correlation
    dataset = Dataset(dataset_name)
    x_clean = drop_constant_cols(dataset.x)
    sorted_idx = np.argsort(np.abs(np.corrcoef(x_clean, y=dataset.y, rowvar=False)[-1, :-1]))
    x_sorted = dataset.x[:, sorted_idx]
    nb_samples, nb_features = x_sorted.shape

    # Nt = nb_features + Nt      # nb of samples for the target dataset
    # Nt = nb_features * 4           # nb of samples for the target dataset
    # Ns = np.max([Nt * 3, 100])      # nb of samples for the source dataset

    df = pd.DataFrame(np.concatenate([x_sorted, dataset.y], axis=1))
    mv = np.arange(nb_features-nb_missing, nb_features)
    pv = nb_features
    Ntst = nb_samples // 10
    n_folds = 30

    # draw random datasets with specified sizes
    data = lin_sim(Nt, Ntst, n_folds, df, mv, pv)
    print("Computing OLS")
    res_ols = [mse(compute_ols(*d[:4]), *d[4:]) for d in tqdm(data)]
    print("Computing non-linear DSFT")
    res_dsftnl = [mse(compute_dsft_nl(*d[:4]), *d[4:]) for d in tqdm(data)]
    print("Computing DSFT")
    res_dsft = [mse(compute_dsft(*d[:4]), *d[4:]) for d in tqdm(data)]
    print("Computing DP")
    res_dp = [mse2(*compute_dp(*d[:4]), *d[4:]) for d in tqdm(data)]

    results = pd.DataFrame()
    results['ols'] = [r[0] for r in res_ols]
    results["dsft"] = [r[0] for r in res_dsft]
    results["dsft_nl"] = [r[0] for r in res_dsftnl]
    results['dp'] = [r[0] for r in res_dp]

    results['dataset'] = dataset_name

    return results


def table_sota_comparison(nb_missing, Nt, datasets, stat_tests=True):
    res = pd.concat([sota_comparison_no_subsample(nb_missing, dataset, Nt=Nt) for dataset in datasets])
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
