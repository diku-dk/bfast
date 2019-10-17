'''
Created on Apr 19, 2018

@author: fgieseke
'''

import copy
import numpy
import pandas
import datetime
from sklearn import linear_model

from ..base import BFAST
from ..utils import check, get_critval

class BFASTCPU(BFAST):

    """ BFAST Monitor implementation using CPUs. The
    interface follows the one of the corresponding R package, 
    see: https://cran.r-project.org/web/packages/bfast   

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
    
    start_monitor : datetime
        A datetime object specifying the start of 
        the monitoring phase.
        
    freq : int, default 365
        The frequency for the seasonal model
    k : int, default 3
        The number of harmonic terms
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
                 ):
        
        super(BFASTCPU, self).__init__(start_monitor,
                                       freq,
                                       k=k,
                                       hfrac=hfrac,
                                       trend=trend,
                                       level=level,
                                       verbose=verbose)
                
    def fit(self, y, dates):
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
        self.n = self._compute_end_history(dates)
        self.lam = self._compute_lam(len(y))
        
        # create (complete) seasonal matrix ("patterns" as columns here!)
        self.mapped_indices = self._map_indices(dates).astype(numpy.int32)

        X = self._create_data_matrix(self.mapped_indices)
                
        # for visualization
        self.y_h = y[:self.n]
        self.y_m = y[self.n:]   
        
        # compute nan mappings
        nans = numpy.isnan(y)
        num_nans = numpy.cumsum(nans)
        val_inds = numpy.array(range(N))[~nans]
        
        # compute new limits (in data NOT containing missing values)
        ns = self.n - num_nans[self.n]
        h = numpy.int(float(ns) * self.hfrac)
        Ns = N - num_nans[N - 1]
        
        if self.verbose > 0:
            print("y={}".format(y))
            print("N={}".format(N))
            print("n={}".format(self.n))
            print("lam={}".format(self.lam))
            print("ns={}".format(ns))
            print("NS={}".format(Ns))
        
        if ns <= 8 or Ns-ns <= 8:
            raise Exception("Not enough observations: ns={ns}, Ns={Ns}".format(ns=ns, Ns=Ns))

        val_inds = val_inds[ns:]
        val_inds -= self.n
        self.val_inds = val_inds 
        
        # remove nan values from patterns+targets
        X_nn = X[:, ~nans]
        y_nn = y[~nans]
        
        # split data into history and monitoring phases
        self.X_nn_h = X_nn[:, :ns]
        self.X_nn_m = X_nn[:, ns:]
        self.y_nn_h = y_nn[:ns]
        self.y_nn_m = y_nn[ns:]
    
        # (1) fit linear regression model for history period
        model = linear_model.LinearRegression()
        model.fit(self.X_nn_h.T, self.y_nn_h)
        
        # for visualization: get predictions for all points
        self.y_pred = model.predict(X.T)
        
        # get predictions for all non-nan points
        y_pred = model.predict(X_nn.T)
        y_error = y_nn - y_pred
    
        # (2) evaluate model on monitoring period mosum_nn process        
        mosum_nn = numpy.empty(Ns - ns)
        for t in range(ns + 1, Ns + 1):
            mosum_nn[t - ns - 1] = y_error[t - h:t].sum()
            if t == ns + 1:
                self.mo_first = y_error[t - h:t].sum()
        self.sigma = numpy.sqrt(numpy.sum(y_error[0:ns] ** 2) / (ns - (2 + 2 * self.k)))
        self.mosum_init = copy.deepcopy(mosum_nn)
        mosum_nn = 1.0 / (self.sigma * numpy.sqrt(ns)) * mosum_nn
        
        self.mosum_nn = mosum_nn
        self.mosum = numpy.empty(N - self.n)
        self.mosum[:] = numpy.nan

        for j in range(len(mosum_nn)):
            idx = val_inds[j]
            self.mosum[idx] = mosum_nn[j]
        
        # boundary and breaks
        self.bounds = self.lam * numpy.sqrt(self._log_plus(self.mapped_indices[self.n:] / numpy.float(self.mapped_indices[-1])))
        self.breaks = numpy.abs(self.mosum) > self.bounds
        
        self.first_break = numpy.where(self.breaks==True)[0]

        if len(self.first_break) > 0:
            self.first_break = self.first_break[0]
        else:
            self.first_break = -1
                
        self.y_error = y_error
        
        return self
    
    def _compute_lam(self, N):
        
        self.period = N / numpy.float(self.n)
        
        check(self.hfrac, self.period, 1 - self.level, "max")
        
        return get_critval(self.hfrac, self.period, 1 - self.level, "max")

    def _compute_end_history(self, dates):
        
        for i in range(len(dates)):
            if self.start_monitor < dates[i]:
                return i 
        
        raise Exception("Date 'start' not within the range of dates!")
        
    def _map_indices(self, dates):
        
        start = dates[0]
        end = dates[-1]
        start = datetime.datetime(start.year, 1, 1)
        end = datetime.datetime(end.year, 12, 31)
        
        drange = pandas.date_range(start, end, freq="d")
        ts = pandas.Series(numpy.ones(len(dates)), dates)
        ts = ts.reindex(drange)
        indices = numpy.argwhere(~numpy.isnan(ts)).T[0]

        return indices
        
    def _create_data_matrix(self, mapped_indices):
        
        N = len(mapped_indices)
        
        temp = 2 * numpy.pi * mapped_indices / numpy.float(self.freq)
    
        if self.trend:
            X = numpy.vstack((numpy.ones(N), mapped_indices))
        else:
            X = numpy.ones(N)
    
        for j in numpy.arange(1, self.k + 1):
            X = numpy.vstack((X, numpy.sin(j * temp)))
            X = numpy.vstack((X, numpy.cos(j * temp)))
 
        return X
    
    def _log_plus(self, a):
        """
        Parameters
        ----------
        a : Numpy 1d array
        """
        
        def f(x):
            if x > numpy.exp(1):
                return numpy.log(x)
            return 1.0
                    
        fv = numpy.vectorize(f, otypes=[numpy.float])
        
        return fv(a)    

    def _store_indices(self, fname):
        
        f = open(fname,'w')
        f.write("[")
        
        elt = [str(a) for a in self.mapped_indices.tolist()]
        line = ",".join(elt)
        f.write(line)
        f.write("]")
            
        f.close()
