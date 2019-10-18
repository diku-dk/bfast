'''
Created on Oct 15, 2019

@author: fgieseke
'''   
    
class BFASTMonitorBase(object):
    
    def __init__(self,
                 start_monitor,
                 freq=365,
                 k=3,
                 hfrac=0.25,
                 trend=True,
                 level=0.05,                 
                 verbose=0,
                 ):
        
        self.start_monitor = start_monitor
        self.freq = freq
        self.k = k
        self.hfrac = hfrac
        self.trend = trend
        self.level = level
        self.verbose = verbose
        
    def fit(self, y):
            
        raise Exception("Function 'fit' not implemented!")    
    