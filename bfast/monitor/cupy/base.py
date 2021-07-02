'''
Created on June 15, 2021

@author: Pierrick Rambaud
'''

import multiprocessing as mp
from functools import partial

import cupy as cp
#cp.warnings.filterwarnings('ignore')
#cp.set_printoptions(suppress=True)
from sklearn import linear_model

from bfast.base import BFASTMonitorBase
from bfast.monitor.cupy_utils import compute_end_history, compute_lam, map_indices


class BFASTMonitorCuPy(BFASTMonitorBase):
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
                 device_id=0
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
        
        # use the specified cuda device (default to 0)
        cp.cuda.device.Device(device_id).use()

    def fit(self, data, dates, nan_value=0, **kwargs):
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
        nan_value : int, default 0
            Specified the NaN value used in
            the array data

        Returns
        -------
        self : instance of BFASTMonitor
            The object itself.
        """
            
        data_ints = cp.copy(data)
        data = cp.array(cp.copy(data_ints)).astype(cp.float32)

        # set NaN values
        data[data_ints==nan_value] = cp.nan

        self.n = compute_end_history(dates, self.start_monitor)

        # create (complete) seasonal matrix ("patterns" as columns here!)
        self.mapped_indices = map_indices(dates).astype(cp.int32)
        self.X = self._create_data_matrix(self.mapped_indices)

        # period = data.shape[0] / cp.float(self.n)
        self.lam = compute_lam(data.shape[0], self.hfrac, self.level, self.period)
        
        results = cp.apply_along_axis(self.fit_single, axis=0, arr=data)
        
        self.breaks = results[0].get()
        self.means = results[1].get()
        self.magnitudes = results[2].get()
        self.valids = results[3].get()

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
        nans = cp.isnan(y)
        num_nans = cp.cumsum(nans)
        val_inds = cp.array(range(N))[~nans]

        # compute new limits (in data NOT containing missing values)
        # ns = n - num_nans[self.n]
        ns = self.n - num_nans[self.n - 1]
        h = cp.int(float(ns) * self.hfrac)
        Ns = N - num_nans[N - 1]

        if ns <= 5 or Ns - ns <= 5:
            brk = -2
            mean = 0.0
            magnitude = 0.0
            if self.verbose > 1:
                print("WARNING: Not enough observations: ns={ns}, Ns={Ns}".format(ns=ns, Ns=Ns))
            
            return cp.array([brk, mean, magnitude, Ns], dtype=cp.float)

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
        coef = cp.linalg.pinv(X_nn_h@X_nn_h.T)@X_nn_h@y_nn_h

        # get predictions for all non-nan points
        y_pred = X_nn.T@coef
        
        y_error = y_nn - y_pred

        # (2) evaluate model on monitoring period mosum_nn process
        err_cs = cp.cumsum(y_error[ns - h:Ns + 1])
        mosum_nn = err_cs[h:] - err_cs[:-h]

        sigma = cp.sqrt(cp.sum(y_error[:ns] ** 2) / (ns - (2 + 2 * self.k)))
        mosum_nn = 1.0 / (sigma * cp.sqrt(ns)) * mosum_nn

        mosum =  cp.full(N - self.n, cp.nan)
        mosum[val_inds[:Ns - ns]] = mosum_nn
        if self.verbose:
            print("MOSUM process", mosum_nn.shape)

        # compute mean
        mean = cp.mean(mosum_nn)

        # compute magnitude
        magnitude = cp.median(y_error[ns:])

        # boundary and breaks
        a = self.mapped_indices[self.n:] / self.mapped_indices[self.n - 1].astype(cp.float)
        bounds = self.lam * cp.sqrt(self._log_plus(a))

        if self.verbose:
            print("lambda", self.lam)
            print("bounds", bounds)

        breaks = cp.abs(mosum) > bounds
        first_break = cp.where(breaks)[0]

        if first_break.shape[0] > 0:
            first_break = first_break[0].item()
        else:
            first_break = -1
        
        return cp.array([first_break, mean.item(), magnitude.item(), Ns.item()], dtype=cp.float)

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
        
        # cast to cp array 
        mapped_indices = cp.array(mapped_indices)
        N = mapped_indices.shape[0]
        temp = 2 * cp.pi * mapped_indices / cp.float(self.freq)

        if self.trend:
            X = cp.vstack((cp.ones(N), mapped_indices))
        else:
            X = cp.ones(N)

        for j in cp.arange(1, self.k + 1):
            X = cp.vstack((X, cp.sin(j * temp)))
            X = cp.vstack((X, cp.cos(j * temp)))

        return X

    def _log_plus(self, a):
        retval = cp.ones(a.shape, dtype=cp.float)
        fl = a > cp.e
        retval[fl] = cp.log(a[fl])

        return retval
