'''
Created on Apr 19, 2018

@author: fgieseke
'''

import calendar
import datetime

import multiprocessing as mp
from functools import partial

import numpy as np
np.warnings.filterwarnings('ignore')
np.set_printoptions(suppress=True)
import pandas
from sklearn import linear_model

from bfast.utils import check, get_critval
from bfast.base import BFASTMonitorBase

class BFASTMonitorPython(BFASTMonitorBase):
    """ BFAST Monitor implementation based on Python and Numpy. The
    interface follows the one of the corresponding R package
    (see https://cran.r-project.org/web/packages/bfast)

    def __init__(self,
                 start_monitor,
                 freq=365,
                 k=3,
                 hfrac=0.25,
                 trend=True,
                 level=0.05,
                 detailed_results=False,
                 old_version=False,
                 verbose=0,
                 platform_id=0,
                 device_id=0
                 ):

    Parameters
    ----------

    start_monitor : datetime object
        A datetime object specifying the start of
        the monitoring phase.

    freq : int, default 365
        The frequency for the seasonal model.

    k : int, default 3
        The number of harmonic terms.

    hfrac : float, default 0.25
        Float in the interval (0,1) specifying the
        bandwidth relative to the sample size in
        the MOSUM/ME monitoring processes.

    trend : bool, default True
        Whether a tend offset term shall be used or not

    level : float, default 0.05
        Significance level of the monitoring (and ROC,
        if selected) procedure, i.e., probability of
        type I error.

    verbose : int, optional (default=0)
        The verbosity level (0=no output, 1=output)

    Examples
    --------

      >>> from bfast import BFASTCPU
      >>> from datetime import datetime
      >>> start_monitor = datetime(2010, 1, 1)
      >>> model = BFASTCPU(start_monitor)

    Notes
    -----

    """

    def __init__(self,
                 start_monitor,
                 freq=365,
                 k=3,
                 hfrac=0.25,
                 trend=True,
                 level=0.05,
                 verbose=0,
                 use_mp=False
                 ):

        super(BFASTMonitorPython, self).__init__(start_monitor,
                                       freq,
                                       k=k,
                                       hfrac=hfrac,
                                       trend=trend,
                                       level=level,
                                       verbose=verbose)

        self._timers = {}
        self.use_mp = use_mp

    def fit(self, data, dates, n_chunks=None, nan_value=0):
        """ Fits the models for the ndarray 'data'

        Parameters
        ----------
        data: ndarray of shape (N, W, H),
            where N is the number of time
            series points per pixel and W
            and H the width and the height
            of the image, respectively.
        dates : list of datetime objects
            Specifies the dates of the elements
            in data indexed by the first axis
            n_chunks : int or None, default None
        nan_value : int, default 0
            Specified the NaN value used in
            the array data

        Returns
        -------
        self : instance of BFASTMonitor
            The object itself.
        """

        data = data.astype(np.float32)

        # set NaN values
        data[data==nan_value] = np.nan

        n = self._compute_end_history(dates)

        # create (complete) seasonal matrix ("patterns" as columns here!)
        mapped_indices = self._map_indices(dates).astype(np.int32)

        X = self._create_data_matrix(mapped_indices)

        if self.use_mp:
            print("Python backend is running in parallel for {} threads".format(mp.cpu_count()))
            my_fun = partial(self.fit_single, X=X, n=n, mapped_indices=mapped_indices)
            y = np.transpose(data, (1, 2, 0)).reshape(data.shape[1] * data.shape[2], data.shape[0])
            pool = mp.Pool(mp.cpu_count())
            rval = pool.map(my_fun, y)
            rval = np.array(rval, dtype=object).reshape(data.shape[1], data.shape[2], 3)

            self.breaks = rval[:,:,0].astype(np.int32)
            self.means = rval[:,:,1].astype(np.float32)
            self.magnitudes = rval[:,:,2].astype(np.float32)
        else:
            means_global = np.zeros((data.shape[1], data.shape[2]), dtype=np.float32)
            magnitudes_global = np.zeros((data.shape[1], data.shape[2]), dtype=np.float32)
            breaks_global = np.zeros((data.shape[1], data.shape[2]), dtype=np.int32)

            for i in range(data.shape[1]):
                if self.verbose > 0:
                    print("Processing row {}".format(i))

                for j in range(data.shape[2]):
                    y = data[:,i,j]
                    pix_break, pix_mean, pix_magnitude = self.fit_single(y, X, n, mapped_indices)
                    breaks_global[i,j] = pix_break
                    means_global[i,j] = pix_mean
                    magnitudes_global[i,j] = pix_magnitude

            self.breaks = breaks_global
            self.means = means_global
            self.magnitudes = magnitudes_global

        return self

    def fit_single(self, y, X, n, mapped_indices):
    # def fit_single(self, y):
        """ Fits the BFAST model for the 1D array y.

        Parameters
        ----------
        y : array
            1d array of length N
        dates : list of datetime objects
            Specifies the dates of the elements
            in data indexed by the first axis

        Returns
        -------
        self : instance of BFASTCPU
            The object itself
        """
        N = len(y)

        period = N / np.float(n)

        lam = self._compute_lam(len(y), period)

        # compute nan mappings
        nans = np.isnan(y)
        num_nans = np.cumsum(nans)
        val_inds = np.array(range(N))[~nans]

        # compute new limits (in data NOT containing missing values)
        # ns = n - num_nans[n]
        ns = n - num_nans[n - 1]
        h = np.int(float(ns) * self.hfrac)
        Ns = N - num_nans[N - 1]

        if ns <= 5 or Ns - ns <= 5:
            brk = -2
            mean = 0.0
            magnitude = 0.0
            if self.verbose > 1:
                print("WARNING: Not enough observations: ns={ns}, Ns={Ns}".format(ns=ns, Ns=Ns))
            return brk, mean, magnitude

        val_inds = val_inds[ns:]
        val_inds -= n

        # remove nan values from patterns+targets
        X_nn = X[:, ~nans]
        y_nn = y[~nans]

        # split data into history and monitoring phases
        X_nn_h = X_nn[:, :ns]
        X_nn_m = X_nn[:, ns:]
        y_nn_h = y_nn[:ns]
        y_nn_m = y_nn[ns:]

        # (1) fit linear regression model for history period
        model = linear_model.LinearRegression(fit_intercept=False)
        model.fit(X_nn_h.T, y_nn_h)

        if self.verbose > 1:
            column_names = np.array(["Intercept",
                                        "trend",
                                        "harmonsin1",
                                        "harmoncos1",
                                        "harmonsin2",
                                        "harmoncos2",
                                        "harmonsin3",
                                        "harmoncos3"])
            if self.trend:
                indxs = np.array([0, 1, 3, 5, 7, 2, 4, 6])
            else:
                indxs = np.array([0, 2, 4, 6, 1, 3, 5])
            print(column_names[indxs])
            print(model.coef_[indxs])

        # get predictions for all non-nan points
        y_pred = model.predict(X_nn.T)
        y_error = y_nn - y_pred

        # (2) evaluate model on monitoring period mosum_nn process
        err_cs = np.cumsum(y_error[ns - h:Ns + 1])
        mosum_nn = err_cs[h:] - err_cs[:-h]

        sigma = np.sqrt(np.sum(y_error[:ns] ** 2) / (ns - (2 + 2 * self.k)))
        mosum_nn = 1.0 / (sigma * np.sqrt(ns)) * mosum_nn

        mosum = np.repeat(np.nan, N - n)
        mosum[val_inds[:Ns - ns]] = mosum_nn

        # copute mean
        mean = np.mean(mosum_nn)

        # compute magnitude
        magnitude = np.median(y_error[ns:])

        # boundary and breaks
        bounds = lam * np.sqrt(self._log_plus(mapped_indices[n:] / np.float(mapped_indices[-1])))
        breaks = np.abs(mosum) > bounds
        first_break = np.where(breaks)[0]

        if len(first_break) > 0:
            first_break = first_break[0]
        else:
            first_break = -1

        return first_break, mean, magnitude

    def get_timers(self):
        """ Returns runtime measurements for the
        different phases of the fitting process.

        Returns
        -------
        dict : An array containing the runtimes
            for the different phases.
        """

        return self._timers

    def _compute_lam(self, N, period):
        check(self.hfrac, period, 1 - self.level, "max")
        return get_critval(self.hfrac, period, 1 - self.level, "max")

    def _compute_end_history(self, dates):
        for i in range(len(dates)):
            # if self.start_monitor < dates[i]:
            if self.start_monitor <= dates[i]:
                return i
        raise Exception("Date 'start' not within the range of dates!")

    def _map_indices(self, dates):

        start = dates[0]
        end = dates[-1]
        start = datetime.datetime(start.year, 1, 1)
        end = datetime.datetime(end.year, 12, 31)

        drange = pandas.date_range(start, end, freq="d")
        ts = pandas.Series(np.ones(len(dates)), dates)
        ts = ts.reindex(drange)
        inds = ~np.isnan(ts.to_numpy())
        indices = np.argwhere(inds).T[0]

        return indices

    def _create_data_matrix(self, mapped_indices):
        N = len(mapped_indices)
        temp = 2 * np.pi * mapped_indices / np.float(self.freq)

        if self.trend:
            X = np.vstack((np.ones(N), mapped_indices))
        else:
            X = np.ones(N)

        for j in np.arange(1, self.k + 1):
            X = np.vstack((X, np.sin(j * temp)))
            X = np.vstack((X, np.cos(j * temp)))

        return X

    def _log_plus(self, a):
        """
        Parameters
        ----------
        a : Np 1d array
        """

        def f(x):
            if x > np.exp(1):
                return np.log(x)
            return 1.0

        fv = np.vectorize(f, otypes=[np.float])

        return fv(a)
