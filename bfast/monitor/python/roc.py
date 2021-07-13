import numpy as np
from scipy import stats, optimize
from bfast.monitor.utils import pval_brownian_motion_max
from .lm import lm

# BFAST strucchange-cpp armadillo version of approx. equal, which is
# an absolute difference |x - y| <= tol.
def approx_equal(x, y, tol):
  return np.mean(np.abs(x - y)) <= tol

def recresid(X, y, tol=None):
    n, k = X.shape
    assert(n == y.shape[0])
    if n == 0:
      return np.array([])

    if tol is None:
        tol = np.sqrt(np.finfo(np.float64).eps) / k

    y = y.reshape(n, 1)
    ret = np.zeros(n - k)

    # initialize recursion
    yh = y[:k] # k x 1
    Xh = X[:k] # k x k
    b, cov_params, rank, _, _ = lm(Xh, yh)

    X1 = cov_params # (X'X)^(-1), k x k
    inds = np.isnan(X1)
    X1[inds] = 0.0
    bhat = np.nan_to_num(b, 0.0) # k x 1

    check = True
    for r in range(k, n):
        prev_rank = rank
        # Compute recursive residual
        x = X[r]
        d = X1 @ x
        fr = 1 + np.dot(x, d)
        resid = y[r] - np.nansum(x * bhat) # dotprod ignoring nans
        ret[r-k] = resid / np.sqrt(fr)

        # Update formulas
        X1 = X1 - (np.outer(d, d))/fr
        bhat += X1 @ x * resid

        # Check numerical stability (rectify if unstable).
        if check:
            # We check update formula value against full OLS fit
            b, cov_params, rank, _, _ = lm(X[:r+1], y[:r+1])
            # R checks nans in fitted parameters; same as rank.
            # Also check on latest recresidual, because fr may
            # be nan.
            nona = (rank == k and prev_rank == k
                              and not np.isnan(ret[r-k]))
            check = not (nona and approx_equal(b, bhat, tol))
            X1 = cov_params
            bhat = np.nan_to_num(b, 0.0)

    return ret

# Recursive CUSUM process
def efp(X, y):
  k, n = X.shape
  w  = recresid(X.T, y)
  sigma = np.std(w, ddof=1)
  process = np.cumsum(np.append([0],w))/(sigma*np.sqrt(n-k))
  return process

# Linear boundary for Brownian motion (limiting process of rec.resid. CUSUM).
def boundary(process, confidence):
    n = process.shape[0]
    t = np.linspace(0, 1, num=n)
    bounds = confidence + (2*confidence*t) # from Zeileis 2002 strucchange paper.
    return bounds

# Structural change test for Brownian motion.
def sctest(process):
    x = process[1:]
    n = x.shape[0]
    j = np.linspace(1/n, 1, num=n)
    x = x * 1/(1 + 2*j)
    stat = np.max(np.abs(x))
    return pval_brownian_motion_max(stat)

def history_roc(X, y, alpha, confidence):
  if y.shape[0] == 0: return 0
  X_rev = np.flip(X, axis=1)
  y_rev = y[::-1]
  rcus = efp(X_rev, y_rev)

  pval = sctest(rcus)
  y_start = 0
  if not np.isnan(pval) and pval < alpha:
      bounds = boundary(rcus, confidence)
      inds = (np.abs(rcus[1:]) > bounds[1:]).nonzero()[0]
      y_start = rcus.size - np.min(inds) - 1 if inds.size > 0 else 0
  return y_start
