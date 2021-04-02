import multiprocessing as mp
from functools import partial

import numpy as np

from recresid import recresid


def ssr_triang(n, h, X, y, k, intercept_only, use_mp=False):
    """
    Calculates the upper triangular matrix of squared residuals
    """
    fun = ssr_triang_par if use_mp else ssr_triang_seq
    return fun(n, h, X, y, k, intercept_only)

def SSRi(i, n, h, X, y, k, intercept_only):
    """
    Compute i'th row of the SSR diagonal matrix, i.e,
    the recursive residuals for segments starting at i = 1:(n-h+1)
    """
    if intercept_only:
        arr1 = np.arange(1, (n-i+1))
        arr2 = arr1[:-1]
        ssr = (y[i:] - np.cumsum(y[i:]) / arr1)[1:] * np.sqrt(1 + 1 / arr2)
    else:
        ssr = recresid(X[i:], y[i:])
        rval = np.concatenate((np.repeat(np.nan, k), np.cumsum(ssr**2)))
        return rval

def ssr_triang_seq(n, h, X, y, k, intercept_only):
    """
    sequential version
    """
    my_SSRi = partial(SSRi, n=n, h=h, X=X, y=y, k=k, intercept_only=intercept_only)
    return np.array([my_SSRi(i) for i in range(n-h+1)], dtype=object)

def ssr_triang_par(n, h, X, y, k, intercept_only):
    """
    parallel version
    """
    my_SSRi = partial(SSRi, n=n, h=h, X=X, y=y, k=k, intercept_only=intercept_only)
    pool = mp.Pool(mp.cpu_count())
    indexes = np.arange(n - h + 1).astype(int)
    rval = pool.map(my_SSRi, indexes)
    rval = np.array(rval, dtype=object)
    return rval
