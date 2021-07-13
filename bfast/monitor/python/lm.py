import numpy as np
from scipy.linalg import cho_solve, solve_triangular

def dqrqty(a, qraux, k, y):
  n, _ = a.shape
  ju = min(k, n-1)

  qty = np.zeros(n)
  qty[0:n] = y[0:n]
  for j in range ( 0, ju ):
    if ( qraux[j] != 0.0 ):
      # put qraux on diagonal
      ajj = a[j,j]
      a[j,j] = qraux[j]
      t = - (np.sum(a[j:n,j] * qty[j:n]))/a[j,j]
      qty[j:n] = qty[j:n] + t*a[j:n,j]
      # revert back to original diagonal
      a[j,j] = ajj
  return qty

def dqrdc2(x, ldx, n, p, tol=1e-7):
  qraux = np.zeros(p)
  work = np.zeros((p,2))
  jpvt = np.arange(p)
  if n > 0:
    for j in range(0,p):
      qraux[j] = np.linalg.norm(x[:,j], 2)
      work[j,0] = qraux[j]
      work[j,1] = qraux[j]
      if work[j,1] == 0.0:
        work[j,1] = 1.0
  lup = min(n,p)
  k = p + 1
  for l in range(0, lup):
    while l+1 < k and qraux[l] < work[l,1]*tol:
      lp1 = l+1
      for i in range(n):
        t = x[i,l]
        for j in range(lp1, p):
          x[i, j-1] = x[i,j]
        x[i,p-1] = t
      i = jpvt[l]
      t = qraux[l]
      tt = work[l,0]
      ttt = work[l,1]
      for j in range(lp1, p):
        jpvt[j-1] = jpvt[j]
        qraux[j-1] = qraux[j]
        work[j-1,:] = work[j,:]
      jpvt[p-1] = i
      qraux[p-1] = t
      work[p-1,0] = tt
      work[p-1,1] = ttt
      k = k -1
    if l+1 != n:
      nrmxl = np.linalg.norm(x[l:, l], 2)
      if nrmxl != 0.0:
        if x[l,l] != 0.0: nrmxl = np.sign(x[l,l]) * abs(nrmxl)
        x[l:, l] = x[l:, l] / nrmxl
        x[l,l] = 1.0 + x[l,l]
        for j in range(l+1, p):
          t = -1*np.dot(x[l:,l], x[l:,j])/x[l,l]
          x[l:,j] = x[l:,j] + t*x[l:,l]
          if qraux[j] != 0.0:
            tt = 1.0 - (abs(x[l,j])/qraux[j])**2
            tt = max(tt, 0.0)
            t = tt
            if abs(t) >= 1e-6:
              qraux[j] = qraux[j]*np.sqrt(t)
            else:
              qraux[j] = np.linalg.norm(x[l+1:,j], 2)
              work[j,0] = qraux[j]
        qraux[l] = x[l,l]
        x[l,l] = -1*nrmxl

  k = min(k-1, n)
  return x, k, qraux, jpvt

def lm(X, y):
  n,p  = X.shape
  y = y.reshape(n)
  A, rank, qraux, jpvt = dqrdc2(X.copy(), n, n, p)
  r = np.triu(A)[:p, :p]
  # Inverting r with cholesky gives (X.T X)^{-1}
  cov_params = cho_solve((r[:rank, :rank], False), np.identity(rank))
  # compute parameters
  qty = dqrqty(A, qraux, rank, y.copy())
  beta = solve_triangular(r[:rank, :rank], qty[:rank])
  # Pivot fitted parameters to match original order of Xs columns
  b = np.zeros(p)
  b[:rank] = beta
  b[jpvt] = b[range(p)]
  scratch = np.zeros((p,p))
  scratch[:rank, :rank] = cov_params
  # swap rows
  scratch[jpvt, :] = scratch[range(p), :]
  # swap columns
  scratch[:, jpvt] = scratch[:, range(p)]
  cov_params = scratch
  return b, cov_params, rank, r, qraux
