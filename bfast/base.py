'''
Created on Oct 15, 2019

@author: fgieseke, mortvest
'''
from abc import ABC
from abc import abstractmethod
from .utils import Logger

class LoggingBase(ABC):
    def __init__(self, verbosity):
        self.verbosity = verbosity
        self.logger = Logger(self.verbosity, self.__class__.__name__)

class BFASTMonitorBase(LoggingBase):
    def __init__(
            self,
            start_monitor,
            freq=365,
            k=3,
            hfrac=0.25,
            trend=True,
            level=0.05,
            period=10,
            verbosity=0,
         ):

        super().__init__(verbosity)
        self.start_monitor = start_monitor
        self.freq = freq
        self.k = k
        self.hfrac = hfrac
        self.trend = trend
        self.level = level
        self.period = period

    @abstractmethod
    def fit(self, y):
        raise Exception("Function 'fit' not implemented!")
