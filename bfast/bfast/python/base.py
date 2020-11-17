'''
Created on Nov 17, 2020


@author: mortvest
'''

import numpy as np

from bfast.base import BFASTBase

class BFASTPython(BFASTBase):
    """
    Python implementation of the BFAST Approach:
    https://www.rdocumentation.org/packages/bfast/versions/1.5.7/topics/bfast

    Parameters
    ----------
    frequency : int

    h : float, default=0.15

    season : string, default="dummy"

    max_iter : int, default=10

    breaks : array, optional

    level : float, default=0.05

    verbose : int, optional (default=0)
        The verbosity level (0=no output, 1=info, 2=debug).


    Attributes
    ----------
    trend : array

    season : array

    remainder : array

    trend_breakpoints : array

    season_breakpoints : array

    """

    def __init__(self,
                 frequency,
                 h=0.15,
                 season_type="dummy",
                 max_iter=10,
                 breaks=None,
                 level=0.05,
                 verbose=0,
                ):
        super(BFASTPython, self).__init__(frequency=frequency,
                                          h=h,
                                          season_type=season_type,
                                          max_iter=max_iter,
                                          breaks=breaks,
                                          level=level,
                                          verbose=verbose)


    def fit(self, Yt, ti):
        """ Fits the models for the ndarray 'data'

        Parameters
        ----------
        Yt: ndarray of shape (N, W, H),
            where N is the number of time
            series points per pixel and W
            and H the width and the height
            of the image, respectively.
        ti : array

        Returns
        -------
        self : The BFAST object.
        """
        print("fitting BFAST Python")
        return self

    def fit_single(self, y, dates):
        """ Fits the BFAST model for the 1D array y.

        Parameters
        ----------
        Yt : array
            1d array of length N
        ti : array

        Returns
        -------
        None
        """
        return self

