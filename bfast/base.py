'''
Created on Oct 15, 2019

@author: fgieseke
'''

from abc import ABC
from abc import abstractmethod


class BFASTMonitorBase(ABC):
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
         ):
        self.start_monitor = start_monitor
        self.freq = freq
        self.k = k
        self.hfrac = hfrac
        self.trend = trend
        self.level = level
        self.period = period
        self.verbose = verbose

    @abstractmethod
    def fit(self, y):
        raise Exception("Function 'fit' not implemented!")
