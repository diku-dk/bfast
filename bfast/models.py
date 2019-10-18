from bfast.cpu import BFASTMonitorCPU
from bfast.gpu import BFASTMonitorGPU

class BFASTMonitor(object):

    """ BFASTMonitor class implementing two different 
    implementations: The first one is a simple, non-optimized
    Python implementation for the BFASTMonitor approach. The
    second one is an efficient massively-parallel implementation. 
    
    The interface follows the one of the corresponding R package, 
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
    backend : str, default 'gpu"
        Specifies the implementation that shall be used:
        backend='cpu' resorts to the non-optimized CPU
        version; backend='gpu' resorts to the optimized
        massively-parallel OpenCL implementation
    platform_id : int, default 0
        Only relevant if backend='gpu'.
        Specifies the OpenCL platform id
    device_id int, default 0
        Only relevant if backend='gpu'.
        Specified the OpenCL device id        
    detailed_results : bool, default False
        Only relevant if backend='gpu'.
        If detailed results should be returned or not
    old_version : bool, default False
        Only relevant if backend='gpu'.
        Specified if an older, non-optimized version
        shall be used

        
    Examples
    --------
    
      >>> from bfast import BFASTMonitor
      >>> from datetime import datetime
      >>> start_monitor = datetime(2010, 1, 1)
      >>> model = BFASTMonitor(start_monitor, backend='gpu')
       
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
                 backend="gpu",
                 platform_id=0,
                 device_id=0,
                 detailed_results=False,
                 old_version=False,
                 ):
        
        self.start_monitor = start_monitor
        self.freq = freq
        self.k = k
        self.hfrac = hfrac
        self.trend = trend
        self.level = level
        self.verbose = verbose
        self.backend = backend
        self.platform_id = platform_id
        self.device_id = device_id
        self.detailed_results = detailed_results
        self.old_version = old_version        
        
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
        Returns : BFASTMonitor
            The object itself
        """
        
        self._model = None

        if self.backend == 'cpu':
            self._model = BFASTMonitorCPU(
                 start_monitor=self.start_monitor,
                 freq=self.freq,
                 k=self.k,
                 hfrac=self.hfrac,
                 trend=self.trend,
                 level=self.level,
                 verbose=self.verbose,
                )
        
        elif self.backend == 'gpu':
            self._model = BFASTMonitorGPU(
                 start_monitor=self.start_monitor,
                 freq=self.freq,
                 k=self.k,
                 hfrac=self.hfrac,
                 trend=self.trend,
                 level=self.level,
                 detailed_results=self.detailed_results,
                 old_version=self.old_version,
                 verbose=self.verbose,
                 platform_id=self.platform_id,
                 device_id=self.device_id
                )
        else:
            
            raise Exception("Unknown backend '{}".format(self.backend))
        
        # fit BFASTMonitor models
        self._model.fit(data=data, 
                        dates=dates, 
                        n_chunks=n_chunks, 
                        nan_value=nan_value)
        self._model_fitted = True
                        
        return self
    
    def get_params(self):
        """ Returns the parameters for this model.
        
        Returns
        -------
        Returns : dict
            Mapping of string to any
            parameter names mapped to their values.
        """
        
        params = {
            "start_monitor": self.start_monitor,
            "freq": self.freq,
            "k": self.k,
            "hfrac": self.hfrac,
            "trend": self.trend,
            "level": self.level,
            "verbose": self.verbose,
            "backend": self.backend,
            "platform_id": self.platform_id,
            "device_id": self.device_id,
            "detailed_results": self.detailed_results,
            "old_version": self.old_version
        }
        
        return params

    def set_params(self, **params):
        """ Sets the parameters for this model.
        
        Parameters
        ----------
        params : dict
            Dictionary containing the 
            parameters to be used.
        """
        
        for parameter, value in params.items():
            self.setattr(parameter, value)
    
    @property
    def timers(self):
        """ Returns runtime measurements for the 
        different phases of the fitting process.

        Returns
        -------
        dict : dict
            An array containing the runtimes 
            for the different phases.
        """        
    
        if self._is_fitted():
            return self._model.get_timers()
        
        raise Exception("Model not yet fitted!")

    @property
    def breaks(self):
        """ Returns the breaks computed

        Returns
        -------
        Returns: array-like
            An array containing the breaks
            computed for the input data
        """        
    
        if self._is_fitted():
            return self._model.breaks
        
        raise Exception("Model not yet fitted!")

    @property
    def means(self):
        """ Returns the means computed

        Returns
        -------
        Returns : array-like
            An array containing the means
            computed for the input data
        """        
    
        if self._is_fitted():
            return self._model.means
        
        raise Exception("Model not yet fitted!")
            
    def _is_fitted(self):

        if hasattr(self, '_model_fitted'):
            return self._model_fitted

        return False