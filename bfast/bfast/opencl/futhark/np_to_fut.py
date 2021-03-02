# converts a .npy 1d array to a 2d futhark array
# usage:
# python np_to_fut.py | futhark dataset -b > data.in

import numpy as np

def add_nans(arr, nan_frac=0.1):
    assert 0 <= nan_frac < 1
    dim = arr.shape[0]
    n_nans = int(dim * nan_frac)
    rng = np.random.default_rng(seed=0)
    idxs = rng.choice(dim, size=n_nans, replace=False)
    arr1 = np.copy(arr)
    arr1[idxs] = np.nan
    return arr1

def num_to_futhark(num, fut_type="f32"):
    if np.isnan(num):
        return fut_type + ".nan"
    return str(num) + fut_type

if __name__ == "__main__":
    # 1d time series location (.npy object)
    ts_loc = "./co2.npy"
    ts = np.load(ts_loc)
    # ts = add_nans(ts)
    formated_single = "[" + ", ".join([num_to_futhark(c) for c in ts]) + "]"
    period = 12
    n_repeats = 10000
    formated_full = "[" + ", ".join([formated_single] * n_repeats) + "]" + " " + "{}i64".format(period)
    print(formated_full)
