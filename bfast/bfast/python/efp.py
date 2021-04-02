import numpy as np
import statsmodels.api as sm

from .utils import omit_nans, sc_me
# from . import datasets


class EFP():
    def __init__(self, X, y, h, verbose=0):
        """
        Empirical fluctuation process. For now, only the Ordinary Least Squares MOving
        SUM (OLS-MOSUM) is supported

        :param X: matrix of x-values
        :param y: vector of y
        :param h: bandwidth parameter for the MOSUM process
        :param deg: degree of the polynomial to be fit [0,1]
        :returns: instance of Empirical Fluctuation Process
        :raises ValueError: wrong type of process
        """
        self.verbose = verbose

        X, y = omit_nans(X, y)
        n, k = X.shape

        if self.verbose > 0:
            print("Performing linear regression")
        # fit linear model
        fm = sm.OLS(y, X, missing='drop').fit()

        e = y - fm.predict(exog=X)
        if self.verbose > 1:
            print("Residuals:\n{}".format(e))

        sigma = np.sqrt(np.sum(e**2) / (n - k))
        if self.verbose > 1:
            print("sigma: {}".format(sigma))

        nh = np.floor(n * h)
        if self.verbose > 1:
            print("nh: {}".format(nh))

        e_zero = np.insert(e, 0, 0)

        process = np.cumsum(e_zero)
        process = process[int(nh):] - process[:(n - int(nh) + 1)]
        process = process / (sigma * np.sqrt(n))
        if self.verbose > 1:
            print("process2:\n{}".format(process))
            print("process1:\n{}".format(process))
            print("process3:\n{}".format(process))

        self.coefficients = fm
        self.sigma = sigma
        self.process = process
        self.par = h

    def p_value(self, x, h, k, max_k=6, table_dim=10):
        """
        Returns the p value for the process.

        :param x: result of application of the functional
        :param h: bandwidth parameter
        :param k: number of rows of matrix X
        :returns: p value for the process
        """
        if self.verbose > 0:
            print("Calculating p-value")

        k = min(k, max_k)

        crit_table = sc_me[((k - 1) * table_dim):(k * table_dim),:]
        tablen = crit_table.shape[1]
        tableh = np.arange(1, table_dim + 1) * 0.05
        tablep = np.array((0.1, 0.05, 0.025, 0.01))
        tableipl = np.zeros(tablen)

        for i in range(tablen):
            tableipl[i] = np.interp(h, tableh, crit_table[:, i])

        if self.verbose > 1:
            print("Interpolated row of p-values:\n{}".format(tableipl))
        tableipl = np.insert(tableipl, 0, 0)
        tablep = np.insert(tablep, 0, 1)

        p = np.interp(x, tableipl, tablep)

        return p

    def sctest(self, functional="max"):
        """
        Performs a generalized fluctuation test.

        :param functional: functional type. Only max is supported
        :raises ValueError: wrong type of functional
        :returns: a tuple of applied functional and p value
        """
        if self.verbose > 0:
            print("Performing statistical test")
        if functional != "max":
            raise ValueError("Functional {} is not supported".format(functional))

        h = self.par
        x = self.process
        if (nd := np.ndim(x)) == 1:
            k = nd
        else:
            k = np.shape[0]

        if self.verbose > 0:
            print("Calculating statistic")
        stat = np.max(np.abs(x))
        if self.verbose > 1:
            print("stat: {}".format(stat))

        p_value = self.p_value(stat, h, k)

        if self.verbose > 1:
            print("p_value: {}".format(stat))

        return (stat, p_value)


def test_dataset(y, name, h=0.15, level=0.15):
    x = np.ones(y.shape[0]).reshape(y.shape[0], 1)
    efp = EFP(x, y, h)
    stat, p_value = efp.sctest()

    print("Testing '{}'".format(name))
    if p_value <= level:
        print("Breakpoint detected")
    else:
        print("No breakpoint detected")

    print("p_value", p_value)
    print("stat", stat)
    print()


if __name__ == "__main__":
    # test_dataset(datasets.nhtemp, "nhtemp")
    test_dataset(datasets.nile, "nile")


