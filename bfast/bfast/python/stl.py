import numpy as np
import matplotlib.pyplot as plt
import statsmodels.tsa.seasonal as sm
import pandas as pd

# import datasets


class STL():
    def __init__(self, y, period, periodic=True):
        y = np.array(pd.DataFrame(y).interpolate().values.ravel().tolist())
        # stl = sm.STL(y, period=period)
        stl = sm.STL(y, period=period)
        res = stl.fit()
        seasonal = res.seasonal
        trend = res.trend
        residual = res.resid
        if periodic:
            seasonal = self.seasonal_average(seasonal, period)

        self.seasonal = seasonal
        self.trend = trend
        self.residual = residual

    def seasonal_average(self, x, period):
        n = x.shape[0]
        use_cut = int(n / period) * period != n
        n_periods = int(n / period)
        if n_periods < 1:
            return x
        if use_cut:
            cut_len = period * n_periods
            mat = x[:cut_len]
        else:
            mat = x
        avg = np.mean(mat.reshape(n_periods, period), axis=0)
        retval = np.tile(avg, n_periods + 1)
        retval = retval[:n]
        return retval


if __name__ == "__main__":
    # Yt = datasets.ndvi
    # ti = datasets.ndvi_dates
    # f = datasets.ndvi_freqency

    # St = STL(Yt, f, periodic=True)
    # print(St.seasonal[99:150])
    from statsmodels.datasets import co2
    import matplotlib.pyplot as plt
    from pandas.plotting import register_matplotlib_converters
    register_matplotlib_converters()
    data = co2.load(True).data

    print(data[:50])
    data = data.resample('M').mean().ffill()

    from statsmodels.tsa.seasonal import STL
    res = STL(data, trend=19, seasonal=35).fit()
    print(np.array(res.trend))
    # res.plot()
    # plt.show()
