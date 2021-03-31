'''
Created on Apr 19, 2018

@author: fgieseke, mortvest
'''

import multiprocessing as mp
from functools import partial

import numpy as np
np.warnings.filterwarnings('ignore')
np.set_printoptions(suppress=True)
from sklearn import linear_model

from bfast.base import BFASTMonitorBase
from bfast.monitor.utils import compute_end_history, compute_lam, map_indices


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

    period : int, default 10
        Maximum time (relative to the history period)
        that will be monitored.

    verbose : int, optional (default=0)
        The verbosity level (0=no output, 1=output)

    use_mp : bool, default False
        Determines whether to use the (very primitive) Python
        multiprocessing or not. Enable for a speedup

    Examples
    --------

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
                 period=10,
                 verbose=0,
                 use_mp=False
                 ):
        super().__init__(start_monitor,
                         freq,
                         k=k,
                         hfrac=hfrac,
                         trend=trend,
                         level=level,
                         period=period,
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
        data_ints = data
        data = np.copy(data_ints).astype(np.float32)

        # set NaN values
        data[data_ints==nan_value] = np.nan

        self.n = compute_end_history(dates, self.start_monitor)

        # create (complete) seasonal matrix ("patterns" as columns here!)
        self.mapped_indices = map_indices(dates).astype(np.int32)
        self.X = self._create_data_matrix(self.mapped_indices)

        # period = data.shape[0] / np.float(self.n)
        self.lam = compute_lam(data.shape[0], self.hfrac, self.level, self.period)

        if self.use_mp:
            print("Python backend is running in parallel using {} threads".format(mp.cpu_count()))
            y = np.transpose(data, (1, 2, 0)).reshape(data.shape[1] * data.shape[2], data.shape[0])
            pool = mp.Pool(mp.cpu_count())
            p_map = pool.map(self.fit_single, y)
            rval = np.array(p_map, dtype=object).reshape(data.shape[1], data.shape[2], 4)

            self.breaks = rval[:,:,0].astype(np.int32)
            self.means = rval[:,:,1].astype(np.float32)
            self.magnitudes = rval[:,:,2].astype(np.float32)
            self.valids = rval[:,:,3].astype(np.int32)
        else:
            means_global = np.zeros((data.shape[1], data.shape[2]), dtype=np.float32)
            magnitudes_global = np.zeros((data.shape[1], data.shape[2]), dtype=np.float32)
            breaks_global = np.zeros((data.shape[1], data.shape[2]), dtype=np.int32)
            valids_global = np.zeros((data.shape[1], data.shape[2]), dtype=np.int32)

            for i in range(data.shape[1]):
                if self.verbose > 0:
                    print("Processing row {}".format(i))

                for j in range(data.shape[2]):
                    y = data[:,i,j]
                    (pix_break,
                     pix_mean,
                     pix_magnitude,
                     pix_num_valid) = self.fit_single(y)
                    breaks_global[i,j] = pix_break
                    means_global[i,j] = pix_mean
                    magnitudes_global[i,j] = pix_magnitude
                    valids_global[i,j] = pix_num_valid

            self.breaks = breaks_global
            self.means = means_global
            self.magnitudes = magnitudes_global
            self.valids = valids_global

        return self

    def fit_single(self, y):
        """ Fits the BFAST model for the 1D array y.

        Parameters
        ----------
        y : array
            1d array of length N

        Returns
        -------
        self : instance of BFASTCPU
            The object itself
        """
        N = y.shape[0]

        # compute nan mappings
        nans = np.isnan(y)
        num_nans = np.cumsum(nans)
        val_inds = np.array(range(N))[~nans]

        # compute new limits (in data NOT containing missing values)
        # ns = n - num_nans[self.n]
        ns = self.n - num_nans[self.n - 1]
        h = np.int(float(ns) * self.hfrac)
        Ns = N - num_nans[N - 1]

        if ns <= 5 or Ns - ns <= 5:
            brk = -2
            mean = 0.0
            magnitude = 0.0
            if self.verbose > 1:
                print("WARNING: Not enough observations: ns={ns}, Ns={Ns}".format(ns=ns, Ns=Ns))
            return brk, mean, magnitude, Ns

        val_inds = val_inds[ns:]
        val_inds -= self.n

        # remove nan values from patterns+targets
        X_nn = self.X[:, ~nans]
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
            # print(column_names[indxs])
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

        mosum = np.repeat(np.nan, N - self.n)
        mosum[val_inds[:Ns - ns]] = mosum_nn
        if self.verbose:
            print("MOSUM process", mosum_nn.shape)

        # compute mean
        mean = np.mean(mosum_nn)

        # compute magnitude
        magnitude = np.median(y_error[ns:])

        # boundary and breaks
        a = self.mapped_indices[self.n:] / self.mapped_indices[self.n - 1].astype(np.float)
        bounds = self.lam * np.sqrt(self._log_plus(a))

        if self.verbose:
            print("lambda", self.lam)
            print("bounds", bounds)

        breaks = np.abs(mosum) > bounds
        first_break = np.where(breaks)[0]

        if first_break.shape[0] > 0:
            first_break = first_break[0]
        else:
            first_break = -1

        return first_break, mean, magnitude, Ns

    def get_timers(self):
        """ Returns runtime measurements for the
        different phases of the fitting process.

        Returns
        -------
        dict : An array containing the runtimes
            for the different phases.
        """
        return self._timers

    def _create_data_matrix(self, mapped_indices):
        N = mapped_indices.shape[0]
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
        retval = np.ones(a.shape, dtype=np.float)
        fl = a > np.e
        retval[fl] = np.log(a[fl])

        return retval
