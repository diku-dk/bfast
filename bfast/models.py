from bfast.monitor import BFASTMonitorPython
from bfast.monitor import BFASTMonitorOpenCL


class BFASTMonitor():
    """
    BFASTMonitor implements the BFASTMonitor approach and
    provides two backends/implementations:

    - A pure Python implementation (based on the Numpy package).
    - An optimized OpenCL implementation suited for massively-parallel devices.

    The interface follows the one of the corresponding R package,
    see: https://cran.r-project.org/web/packages/bfast

    Parameters
    ----------
    start_monitor : datetime
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
        Whether a tend offset term shall be used or not.

    level : float, default 0.05
        Significance level of the monitoring (and ROC,
        if selected) procedure, i.e., probability of
        type I error.

    verbose : int, optional (default=0)
        The verbosity level (0=no output, 1=output).

    backend : str, default 'opencl"
        Specifies the implementation that shall be used:
        backend='python' resorts to the non-optimized Python
        version; backend='opencl' resorts to the optimized
        massively-parallel OpenCL implementation.

    platform_id : int, default 0
        Only relevant if backend='opencl'.
        Specifies the OpenCL platform id.

    device_id int, default 0
        Only relevant if backend='opencl'.
        Specified the OpenCL device id.

    detailed_results : bool, default False
        Only relevant if backend='opencl'.
        If detailed results should be returned or not.

    find_magnitudes : bool, default True
        Only relevant if backend='opencl'.
        If magnitudes should be returned or not.
        Disabling this would improve the performance greatly


    Attributes
    ----------
    breaks : array
        An array containing the (first) break detected. A '-2'
        corresponds to pixels (time series) not containing
        sufficient data and a '-1' to pixels without any breaks.
        All other non-negative entries correspond to the first
        break that was detected in the monitor period (i.e.,
        its index).

    means : array
        An array containing the mean values of the individual
        MOSUM processes (e.g., a positive mean for a pixel
        corresponds to an increase of the vegetation in case
        indices such as NDMI are considered).

    timers : dict
        An dictionary containing the runtimes  for the
        different phases.

    """
    def __init__(
            self,
            start_monitor,
            freq=365,
            k=3,
            hfrac=0.25,
            trend=True,
            level=0.05,
            period=10,
            verbose=0,
            backend="opencl",
            platform_id=0,
            device_id=0,
            detailed_results=False,
            find_magnitudes=True,
        ):
        self.start_monitor = start_monitor
        self.freq = freq
        self.k = k
        self.hfrac = hfrac
        self.trend = trend
        self.level = level
        self.period = period
        self.verbose = verbose
        self.backend = backend
        self.platform_id = platform_id
        self.device_id = device_id
        self.detailed_results = detailed_results
        self.find_magnitudes = find_magnitudes

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
        self : The BFASTMonitor object.
        """
        self._model = None

        if self.backend == 'python':
            self._model = BFASTMonitorPython(
                 start_monitor=self.start_monitor,
                 freq=self.freq,
                 k=self.k,
                 hfrac=self.hfrac,
                 trend=self.trend,
                 level=self.level,
                 period=self.period,
                 verbose=self.verbose,
                )

        elif self.backend == 'python-mp':
            self._model = BFASTMonitorPython(
                 start_monitor=self.start_monitor,
                 freq=self.freq,
                 k=self.k,
                 hfrac=self.hfrac,
                 trend=self.trend,
                 level=self.level,
                 period=self.period,
                 verbose=self.verbose,
                 use_mp=True
                )

        elif self.backend == 'opencl':
            self._model = BFASTMonitorOpenCL(
                 start_monitor=self.start_monitor,
                 freq=self.freq,
                 k=self.k,
                 hfrac=self.hfrac,
                 trend=self.trend,
                 level=self.level,
                 period=self.period,
                 detailed_results=self.detailed_results,
                 find_magnitudes=self.find_magnitudes,
                 verbose=self.verbose,
                 platform_id=self.platform_id,
                 device_id=self.device_id
                )

        elif self.backend == "C":
            raise Exception("Multi-core C implementation will be added soon!")

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
        dict: Mapping of string to any
            parameter names mapped to their values.
        """
        params = {
            "start_monitor": self.start_monitor,
            "freq": self.freq,
            "k": self.k,
            "hfrac": self.hfrac,
            "trend": self.trend,
            "level": self.level,
            "period": self.period,
            "verbose": self.verbose,
            "backend": self.backend,
            "platform_id": self.platform_id,
            "device_id": self.device_id,
            "detailed_results": self.detailed_results,
            "find_magnitudes": self.find_magnitudes
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
        dict : An dictionary containing the runtimes
            for the different phases.
        """
        if self._is_fitted():
            return self._model.get_timers()

        raise Exception("Model not yet fitted!")

    @property
    def breaks(self):
        """ Returns the breaks that have been computed.

        Returns
        -------
        array-like: An array containing the (first) breaks
            detected by BFASTMonitor. A '-2' corresponds
            to pixels (time series) not containing sufficient data
            and a '-1' to pixels without any breaks. All other non-negative
            entries correspond to the first break that was
            detected in the monitor period (i.e., its index).
        """
        if self._is_fitted():
            return self._model.breaks

        raise Exception("Model not yet fitted!")

    @property
    def means(self):
        """ Returns the means computed

        Returns
        -------
        array-like : An array containing the mean values
            of the individual MOSUM processes (e.g., a positive
            mean for a pixel corresponds to an increase of the
            vegetation in case indices such as NDMI are considered)
        """
        if self._is_fitted():
            return self._model.means

        raise Exception("Model not yet fitted!")

    @property
    def magnitudes(self):
        """ Returns the magnitudes computed

        Returns
        -------
        array-like : An array containing the magnitude
            values median of the difference between the data
            and the model prediction in the monitoring period
        """
        if self._is_fitted():
            return self._model.magnitudes

        raise Exception("Model not yet fitted!")

    def _is_fitted(self):
        if hasattr(self, '_model_fitted'):
            return self._model_fitted

        return False

    @property
    def valids(self):
        """ Returns the number of valid values for each pixel

        Returns
        -------
         array-like : An array containing the number
             of valid values for each pixel in the
             aray data
        """
        if self._is_fitted():
            return self._model.valids

        raise Exception("Model not yet fitted!")
