import numpy as np
import statsmodels.api as sm


def _no_nans(arr):
    return not np.isnan(arr).any()

def _Xinv0(x, coeffs):
    """
    Approximate (X'X)^-1 using QR decomposition
    """
    ncol = np.shape(x)[1]
    r = np.linalg.qr(x)[1]
    qr_rank = np.linalg.matrix_rank(r)

    r = r[:qr_rank, :qr_rank]

    k = coeffs.shape[0]
    rval = np.zeros((k, k))
    rval[:qr_rank, :qr_rank] = np.linalg.inv(r.T @ r)
    return rval

def recresid(x, y, start=None, end=None, tol=None):
    """
    Function for computing the recursive residuals (standardized one step
    prediction errors) of a linear regression model.
    """
    if np.ndim(x) == 1:
        ncol = 1
        nrow = x.shape
    else:
        nrow, ncol = x.shape

    if start is None:
        start = ncol + 1
    if end is None:
        end = nrow
    if tol is None:
        tol = np.sqrt(np.finfo(float).eps / ncol)

    # checks and data dimensions
    assert start > ncol and start <= nrow
    assert end >= start and end <= nrow

    n = end
    q = start - 1
    k = ncol
    rval = np.zeros(n - q)

    # initialize recursion
    y1 = y[:q]

    x_q = x[:q]
    # coeffs = np.polyfit(x_q.flatten(), y1, deg=deg)

    model = sm.OLS(y1, x_q, missing='drop').fit()
    coeffs = model.params

    X1 = _Xinv0(x_q, coeffs)
    betar = np.nan_to_num(coeffs)

    xr = x[q]
    fr = 1 + (xr @ X1 @ xr)
    rval[0] = (y[q] - xr @ betar) / np.sqrt(fr)

    # check recursion against full QR decomposition?
    check = True

    if (q + 1) >= n:
        return rval

    for r in range(q + 1, n):
        # check for NAs in coefficients
        nona = _no_nans(coeffs)

        # recursion formula
        X1 = X1 - (X1 @ np.outer(xr, xr) @ X1)/fr
        # print("X1", X1)
        betar += X1 @ xr * rval[r-q-1] * np.sqrt(fr)
        # print("betar", betar)

        # full QR decomposition
        if check:
            y1 = y[:r]
            # print("y1", y1)
            x_i = x[:r]
            # print("x_i", x_i)
            model = sm.OLS(y1, x_i, missing='drop').fit()
            coeffs = model.params
            # print("coeffs", coeffs)
            nona = nona and _no_nans(betar) and _no_nans(coeffs)

            # keep checking?
            check = not (nona and np.allclose(coeffs, betar, atol=tol))
            X1 = _Xinv0(x_i, coeffs)
            # print("check_X1", X1)
            betar = np.nan_to_num(coeffs)

        # residual
        xr = x[r]
        fr = 1 + xr @ X1 @ xr
        val = np.nan_to_num(xr * betar)
        v = (y[r] - np.sum(val)) / np.sqrt(fr)
        rval[r-q] = v

    # exit()
    rval = np.around(rval, 8)
    return rval


if __name__ == "__main__":
    x = np.arange(1,21)
    y = x * 2
    print(y)
    y[9:] = y[9:] + 10
    X = sm.add_constant(x)

    rec_res = recresid(X, y)
    print(rec_res)
