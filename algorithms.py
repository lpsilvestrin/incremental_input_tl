import numpy as np
from sklearn.metrics.pairwise import rbf_kernel


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


if __name__ == '__main__':
    def test1():
        xs = np.random.randn(10, 4)
        xtc = np.random.randn(5, 4)
        xt = np.random.randn(5, 2)
        wt = dsft_rbf(xs, xtc, xt)
        print(wt)
