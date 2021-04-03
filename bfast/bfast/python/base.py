'''
Created on Nov 17, 2020


@author: mortvest
'''

import numpy as np
import statsmodels.api as sm

from bfast.base import BFASTBase
from . import utils
from .stl import STL
from .efp import EFP
from .breakpoints import Breakpoints


class BFASTPython(BFASTBase):
    """
    Python implementation of the BFAST algorithm:
    https://www.rdocumentation.org/packages/bfast/versions/1.5.7/topics/bfast

    Iterative break detection in seasonal and trend component of a time
    series. Seasonal breaks is a function that combines the iterative
    decomposition of time series into trend, seasonal and remainder
    components with significant break detection in the decomposed
    components of the time series.

    Parameters
    ----------
    frequency : int

    h : float, default=0.15

    season : string, default="dummy"

    max_iter : int, default=10

    max_breaks : int, optional

    level : float, default=0.05

    verbose : int, optional (default=0)
        The verbosity level (0=no output, 1=info, 2=debug).


    Attributes
    ----------
    trend : ndarray

    season : ndarray

    remainder : ndarray

    trend_breakpoints : ndarray

    season_breakpoints : ndarray

    n_trend_breakpoints : ndarray

    n_season_breakpoints : ndarray

    """

    def __init__(self,
                 frequency,
                 h=0.15,
                 season_type="dummy",
                 max_iter=10,
                 max_breaks=None,
                 level=0.05,
                 verbose=0,
                ):
        super(BFASTPython, self).__init__(frequency=frequency,
                                          h=h,
                                          season_type=season_type,
                                          max_iter=max_iter,
                                          max_breaks=max_breaks,
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
        trend_global = np.zeros(Yt.shape, dtype=np.float32)
        season_global = np.zeros(Yt.shape, dtype=np.float32)
        remainder_global = np.zeros(Yt.shape, dtype=np.float32)
        trend_breakpoints_global = np.zeros(Yt.shape, dtype=int)
        season_breakpoints_global = np.zeros(Yt.shape, dtype=int)
        n_trend_breakpoints_global = np.zeros((Yt.shape[1], Yt.shape[2]), dtype=int)
        n_season_breakpoints_global = np.zeros((Yt.shape[1], Yt.shape[2]), dtype=int)

        for i in range(Yt.shape[1]):
            if self.verbose > 0:
                print("Processing row {}".format(i))
            for j in range(Yt.shape[2]):
                y = Yt[:,i,j]
                pix_trend, \
                    pix_season, \
                    pix_remainder, \
                    pix_trend_br, \
                    pix_season_br, \
                    pix_n_trend_br, \
                    pix_n_season_br = self.fit_single(y, ti)

                trend_global[:, i, j] = pix_trend
                season_global[:, i, j] = pix_season
                remainder_global[:, i, j] = pix_remainder
                trend_breakpoints_global[:, i, j] = pix_trend_br
                season_breakpoints_global[:, i, j] = pix_season_br
                n_trend_breakpoints_global[i, j] = pix_n_trend_br
                n_season_breakpoints_global[i, j] = pix_n_season_br

        self.trend = trend_global
        self.season = season_global
        self.remainder = remainder_global
        self.trend_breakpoints = trend_breakpoints_global
        self.season_breakpoints = season_breakpoints_global
        self.n_trend_breakpoints = n_trend_breakpoints_global
        self.n_season_breakpoints = n_season_breakpoints_global

        return self

    def fit_single(self, Yt, ti):
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
        nrow = Yt.shape[0]
        Tt = None
        f = self.frequency

        if self.season_type == "harmonic":
            if self.verbose > 0:
                print("'harmonic' season is chosen")
            w = 1/f
            tl = np.arange(1, Yt.shape[0] + 1)
            co = np.cos(2 * np.pi * tl * w)
            si = np.sin(2 * np.pi * tl * w)
            co2 = np.cos(2 * np.pi * tl * w * 2)
            si2 = np.sin(2 * np.pi * tl * w * 2)
            co3 = np.cos(2 * np.pi * tl * w * 3)
            si3 = np.sin(2 * np.pi * tl * w * 3)
            smod = np.column_stack((co, si, co2, si2, co3, si3))

            # Start the iterative procedure and for first iteration St=decompose result
            if self.verbose > 0:
                print("Applying STL")
            St = STL(Yt, f, periodic=True).seasonal
            if self.verbose > 1:
                print("St set to\n{}".format(St))
        elif self.season_type == "dummy":
            if self.verbose > 0:
                print("'dummy' season is chosen")
            # Start the iterative procedure and for first iteration St=decompose result
            if self.verbose > 0:
                print("Applying STL")
            St = STL(Yt, f, periodic=True).seasonal

            eye_box = np.row_stack((np.eye(f - 1), np.repeat(-1, f - 1)))
            n_boxes = int(np.ceil(nrow / f))

            smod = np.tile(eye_box, (n_boxes, 1))
            smod = smod[:nrow]
            smod = sm.add_constant(smod)
        elif self.season_type == "none":
            if self.verbose > 0:
                print("'none' season is chosen")
                print("No sesonal model will be fitted!")
            St = np.zeros(nrow)
        else:
            raise ValueError("Seasonal model is unknown, use 'harmonic', 'dummy' or 'none'")

        Vt_bp = np.array([0])
        Wt_bp = np.array([0])
        CheckTimeTt = np.array([1])
        CheckTimeSt = np.array([1])
        i_iter = 1
        nan_map = utils.nan_map(Yt)

        while (Vt_bp != CheckTimeTt).any() or (Wt_bp != CheckTimeSt).any() and i_iter < max_iter:
            if self.verbose > 0:
                print("BFAST iteration #{}".format(i_iter))
            CheckTimeTt = Vt_bp
            CheckTimeSt = Wt_bp

            ### Change in trend component
            with np.errstate(invalid="ignore"):
                Vt = Yt - St  # Deseasonalized Time series
            if self.verbose > 1:
                print("Vt:\n{}".format(Vt))
            p_Vt = EFP(sm.add_constant(ti), Vt, self.h, verbose=self.verbose).sctest()
            if p_Vt[1] <= self.level:
                if self.verbose > 0:
                    print("Breakpoints in trend detected")
                ti1, Vt1 = utils.omit_nans(ti, Vt)
                if self.verbose > 0:
                    print("Finding breakpoints in trend")
                bp_Vt = Breakpoints(sm.add_constant(ti1),
                                    Vt1, h=self.h,
                                    max_breaks=self.max_breaks,
                                    verbose=self.verbose)
                if bp_Vt.breakpoints is not None:
                    bp_Vt.breakpoints_no_nans = np.array([nan_map[i] for i in bp_Vt.breakpoints])
                    nobp_Vt = False
                else:
                    nobp_Vt = True
            else:
                if self.verbose > 0:
                    print("No breakpoints in trend detected")
                nobp_Vt = True
                bp_Vt = None

            if self.verbose > 0:
                print("Fitting linear model for trend")
            if nobp_Vt:
                ## No Change detected
                fm0 = sm.OLS(Vt, ti, missing='drop').fit()
                Vt_bp = np.array([0])  # no breaks times

                Tt = fm0.predict(exog=ti)  # Overwrite non-missing with fitted values
                Tt[np.isnan(Yt)] = np.nan
            else:
                part = bp_Vt.breakfactor()
                X1 = utils.partition_matrix(part, sm.add_constant(ti[~np.isnan(Yt)]))
                y1 = Vt[~np.isnan(Yt)]

                fm1 = sm.OLS(y1, X1, missing='drop').fit()
                Vt_bp = bp_Vt.breakpoints

                Tt = np.repeat(np.nan, ti.shape[0])
                Tt[~np.isnan(Yt)] = fm1.predict()

            if self.season_type == "none":
                Wt = np.zeros(nrow).astype(float)
                St = np.zeros(nrow).astype(float)
                bp_Wt = None
                nobp_Wt = True
            else:
                ### Change in seasonal component
                with np.errstate(invalid="ignore"):
                    Wt = Yt - Tt
                p_Wt = EFP(smod, Wt, self.h, verbose=self.verbose).sctest()  # preliminary test
                if p_Wt[1] <= self.level:
                    if self.verbose > 0:
                        print("Breakpoints in season detected")
                    smod1, Wt1 = utils.omit_nans(smod, Wt)
                    if self.verbose > 0:
                        print("Finding breakpoints in season")
                    bp_Wt = Breakpoints(smod1,
                                        Wt1,
                                        h=self.h,
                                        max_breaks=self.max_breaks,
                                        verbose=self.verbose)
                    if bp_Wt.breakpoints is not None:
                        bp_Wt.breakpoints_no_nans = np.array([nan_map[i] for i in bp_Wt.breakpoints])
                        nobp_Wt = False
                    else:
                        nobp_Wt = True
                else:
                    if self.verbose > 0:
                        print("No breakpoints in season detected")
                    nobp_Wt = True
                    bp_Wt = None

                if self.verbose > 0:
                    print("Fitting linear model for season")
                if nobp_Wt:
                    ## No seasonal change detected
                    sm0 = sm.OLS(Wt, smod, missing='drop').fit()
                    St = np.repeat(np.nan, nrow)
                    St[~np.isnan(Yt)] = sm0.predict()  # Overwrite non-missing with fitted values
                    Wt_bp = np.array([0])
                else:
                    part = bp_Wt.breakfactor()
                    if season in ["dummy", "harmonic"]:
                        X_sm1 = utils.partition_matrix(part, smod1)

                    sm1 = sm.OLS(Wt1, X_sm1, missing='drop').fit()
                    # Wt_bp = bp_Wt.breakpoints_no_nans
                    Wt_bp = bp_Wt.breakpoints

                    # Define empty copy of original time series
                    St = np.repeat(np.nan, nrow)
                    St[~np.isnan(Yt)] = sm1.predict()  # Overwrite non-missing with fitted values

            with np.errstate(invalid="ignore"):
                Nt = Yt - Tt - St

            i_iter += 1

        Vt_bp_arr = np.zeros(nrow)
        Vt_bp_arr[:len(Vt_bp)] = np.array(Vt_bp)
        Wt_bp_arr = np.zeros(nrow)
        Wt_bp_arr[:len(Wt_bp)] = np.array(Wt_bp)

        return Tt, St, Nt, Vt_bp_arr, Wt_bp_arr, len(Vt_bp), len(Wt_bp)
