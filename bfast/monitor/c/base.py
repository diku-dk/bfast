'''
Created on 18.10.2019

@author: fgieseke
'''

import numpy
from bfast.utils import check, get_critval
from bfast.base import BFASTMonitorBase

class BFASTMonitorC(BFASTMonitorBase):

    """ BFAST Monitor implementation based on C. The
    interface follows the one of the corresponding R package, 
    see: https://cran.r-project.org/web/packages/bfast   

                         
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
        
        super(BFASTMonitorC, self).__init__(start_monitor,
                                       freq,
                                       k=k,
                                       hfrac=hfrac,
                                       trend=trend,
                                       level=level,
                                       verbose=verbose)
        
        self._timers = {}
        
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
            The object itself
        """

        data = data.astype(numpy.float32)
        
        raise Exception("Will be added soon!")