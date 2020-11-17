import logging

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from setup import logging_setup
import datasets
from ssr_triang import ssr_triang


logger = logging.getLogger(__name__)


class Breakpoints():
    def __init__(self, X, y, h=0.15, breaks=None, use_mp=False):
        """
        Computation of optimal breakpoints in regression relationships.

        :param X: matrix of x-values
        :param y: vector of y
        :param h: minimum segment width (0<h<1) as fraction of input length
        :param breaks: maximum number of breakpoints (optional)
        :returns: instance of Breakpoints
        """
        y = np.array(pd.DataFrame(y).interpolate().values.ravel().tolist())
        n, k = X.shape
        self.nobs = n
        logger.debug("n = {}, k = {}".format(n, k))

        intercept_only = np.allclose(X, 1)
        logger.debug("intercept_only = {}".format(intercept_only))

        h = int(np.floor(n * h))
        self.h = h
        logger.debug("h = {}".format(h))

        max_breaks = int(np.ceil(n / h) - 2)
        if breaks is None:
            breaks = max_breaks
        elif breaks > max_breaks:
            logger.warning("requested number of breaks = {} too large, changed to {}".
                            format(breaks, max_breaks))
            breaks = max_breaks

        logger.debug("breaks = {}".format(breaks))
        ## compute optimal previous partner if observation i is the mth break
        ## store results together with SSRs in SSR_table


        logger.info("Calculating triangular matrix")
        # self.SSR_triang = \
        #     np.array([SSRi(i) for i in np.arange(n-h+1).astype(int)], dtype=object)

        self.SSR_triang = ssr_triang(n, h, X, y, k, intercept_only, use_mp=use_mp)

        logger.debug("SSR_triang:\n{}".format(self.SSR_triang))


        index = np.arange((h - 1), (n - h)).astype(int)
        logger.debug("index:\n{}".format(index))

        ## 1 break
        break_SSR = np.array([self.SSR(0, i) for i in index])
        logger.debug("break_SSR:\n{}".format(break_SSR))

        SSR_table = np.column_stack((index, break_SSR))
        logger.debug("SSR_table:\n{}".format(SSR_table))

        ## breaks >= 2
        SSR_table = self.extend_SSR_table(SSR_table, breaks)
        logger.debug("extended SSR_table:\n{}".format(SSR_table))

        opt = self.extract_breaks(SSR_table, breaks).astype(int)
        self.breakpoints = opt

        self.SSR_table = SSR_table
        self.nreg = k
        self.y = y
        self.X = X

        # find the optimal amount of breakpoints using Bayesian Information Criterion
        bp = self.breakpoints_for_m()[1]

        if bp is None:
            self.breakpoints_no_nans = bp
            self.breakpoints = bp
        else:
            self.breakpoints_no_nans = bp.astype(int)
            self.breakpoints = bp.astype(int)

    def SSR(self, i, j):
        return self.SSR_triang[int(i)][int(j - i)]

    def extend_SSR_table(self, SSR_table, breaks):
        _, ncol = SSR_table.shape
        h = self.h
        n = self.nobs

        if (breaks * 2) > ncol:
            v1 = int(ncol/2) + 1
            v2 = breaks

            if v1 < v2:
                loop_range = np.arange(v1, v2 + 1)
            else:
                loop_range = np.arange(v1, v2 - 1, -1)

            for m in loop_range:
                my_index = np.arange((m * h) - 1, (n - h))
                index_arr = np.arange((m-1)*2 - 2, (m-1)*2)
                my_SSR_table = SSR_table[:, index_arr]
                nans = np.repeat(np.nan, my_SSR_table.shape[0])
                my_SSR_table = np.column_stack((my_SSR_table, nans, nans))
                for i in my_index:
                    pot_index = np.arange((m - 1) * h - 1, (i - h + 1)).astype(int)
                    fun = lambda j: my_SSR_table[j - h + 1, 1] + self.SSR(j + 1, i)
                    break_SSR = np.vectorize(fun)(pot_index)
                    opt = np.nanargmin(break_SSR)
                    my_SSR_table[i - h + 1, np.array((2, 3))] = \
                        np.array((pot_index[opt], break_SSR[opt]))
                SSR_table = np.column_stack((SSR_table, my_SSR_table[:, np.array((2,3))]))
        return(SSR_table)

    def extract_breaks(self, SSR_table, breaks):
        """
        extract optimal breaks
        """
        _, ncol = SSR_table.shape
        n = self.nobs
        h = self.h

        if (breaks * 2) > ncol:
            raise ValueError("compute SSR_table with enough breaks before")

        index = SSR_table[:, 0].astype(int)
        fun = lambda i: SSR_table[int(i - self.h + 1), int(breaks * 2 - 1)] \
            + self.SSR(i + 1, n - 1)
        break_SSR = np.vectorize(fun)(index)
        opt = [index[np.nanargmin(break_SSR)]]

        if breaks > 1:
            for i in 2 * np.arange(breaks, 1, -1).astype(int) - 2:
                opt.insert(0, SSR_table[int(opt[0] - h + 1), i])
        return(np.array(opt))

    def breakpoints_for_m(self, breaks=None):
        logger.info("running breakpoints for m = {}".format(breaks))
        if breaks is None:
            sbp = self.summary()
            # breaks = np.argmin(sbp[1]) - 1
            breaks = np.argmin(sbp[1])
            logger.debug("BIC:\n{}".format(sbp[1]))
        if breaks < 1:
            SSR = self.SSR(0, self.nobs - 1)
            return SSR, None
        else:
            SSR_tab = self.extend_SSR_table(self.SSR_table, breaks)
            breakpoints = self.extract_breaks(SSR_tab, breaks)
            bp = np.concatenate(([0], breakpoints, [self.nobs-1]))
            cb = np.column_stack((bp[:-1] + 1, bp[1:]))
            fun = lambda x: self.SSR(x[0], x[1])
            SSR = np.sum([fun(i) for i in cb])
            return SSR, breakpoints

    def summary(self, breaks=None, sort=True, format_times=None):
        if breaks is None:
            breaks = int(self.SSR_table.shape[1]/2)

        n = self.nobs
        SSR = np.concatenate(([self.SSR(0, n-1)], np.repeat(np.nan, breaks)))
        if np.isclose(SSR[0], 0.0):
            BIC_vals = -np.inf
        else:
            BIC_vals = n * (np.log(SSR[0]) + 1 - np.log(n) + np.log(2*np.pi)) \
                + np.log(n) * (self.nreg + 1)
        BIC = np.concatenate(([BIC_vals], np.repeat(np.nan, breaks)))
        SSR1, breakpoints = self.breakpoints_for_m(breaks)
        SSR[breaks] = SSR1
        BIC[breaks] = self.BIC(SSR1, breakpoints)

        if breaks > 1:
            for m in range(breaks - 1, 0, -1):
                SSR_m, breakpoints_m = self.breakpoints_for_m(breaks=m)
                SSR[m] = SSR_m
                BIC[m] = self.BIC(SSR_m, breakpoints_m)
        SSR = np.vstack((SSR, BIC))
        return SSR

    def BIC(self, SSR, breakpoints):
        """
        Bayesian Information Criterion
        """
        if np.isclose(SSR, 0.0):
            return -np.inf
        n = self.nobs
        bp = breakpoints
        df = (self.nreg + 1) * (len(bp[~np.isnan(bp)]) + 1)
        # log-likelihood
        logL = n * (np.log(SSR) + 1 - np.log(n) + np.log(2 * np.pi))
        bic = df * np.log(n) + logL
        return bic

    def breakfactor(self):
        logger.info("running breakfactor")
        breaks = self.breakpoints
        nobs = self.nobs
        if np.isnan(breaks).all():
            return np.repeat(1, nobs)

        nbreaks = breaks.shape[0]
        v = np.insert(np.diff(np.append(breaks, nobs)), 0, breaks[0]).astype(int)
        fac = np.repeat(np.arange(1, nbreaks + 2), v)
        return fac - 1


if __name__ == "__main__":
    logging_setup()
    logger = logging.getLogger(__name__)

    print("Testing synthetic")
    # Synthetic dataset with two breakpoints x = 15 and 35
    n = 50
    ones = np.ones(n).reshape((n, 1)).astype("float64")
    y = np.arange(1, n+1).astype("float64")
    X = np.copy(y).reshape((n, 1))
    # X = np.column_stack((ones, X))
    # X = ones
    # X[5] = np.nan
    y[14:] = y[14:] * 0.03
    y[5] = np.nan
    y[34:] = y[34:] + 10


    bp = Breakpoints(X, y, use_mp=False).breakpoints
    print("Breakpoints:", bp)
    print()


    # # Nile dataset with a single breakpoint. Ashwan dam was built in 1898
    # print("Testing nile")

    # y = nile
    # X = np.ones(y.shape[0]).reshape((y.shape[0], 1))

    # bp_nile = Breakpoints(X, y)
    # bp_nile_arr = bp_nile.breakpoints
    # print(bp_nile_arr)

    # BUG v
    # bp_nile_bf = bp_nile.breakfactor()[0]

    # # plt.plot(nile_dates[bp_nile_bf == 1], nile[bp_nile_bf == 1])
    # # plt.plot(nile_dates[bp_nile_bf == 2], nile[bp_nile_bf == 2])
    # # plt.show()

    # nile_break_date = nile_dates[bp_nile_arr]
    # print("Breakpoints:", nile_break_date)
    # print()

    # # UK Seatbelt data. Has at least two break points: one in 1973 and one in 1983
    # print("Testing UK Seatbelt data")

    # y = uk_driver_deaths
    # X = np.ones(y.shape[0]).reshape((y.shape[0], 1))
    # uk_breaks = 2

    # # plt.plot(uk_driver_deaths_dates, uk_driver_deaths)
    # # plt.show()

    # bp_uk = Breakpoints(X, y, breaks=uk_breaks).breakpoints
    # uk_break_dates = uk_driver_deaths_dates[bp_uk]
    # if uk_break_dates.shape[0] > 0:
    #     print("Breakpoints", uk_break_dates)
