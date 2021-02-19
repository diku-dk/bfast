# python np_to_fut.py | futhark dataset -b > data.in
import numpy as np

def add_nans(arr):
    dim = arr.shape[0]
    rng = np.random.default_rng(seed=0)
    idxs = rng.choice(dim, size=40, replace=False)
    arr1 = np.copy(arr)
    arr1[idxs] = np.nan
    return arr1

def num_to_futhark(num, fut_type="f32"):
    if np.isnan(num):
        return fut_type + ".nan"
    return str(num) + fut_type

if __name__ == "__main__":
    co2 = np.load("co2.npy")
    # co2 = add_nans(co2)
    formated_single = "[" + ", ".join([num_to_futhark(c) for c in co2]) + "]"
    period = 12
    # n_repeats = 500 * 500
    n_repeats = 10000
    formated_full = "[" + ", ".join([formated_single] * n_repeats) + "]" + " " + "{}i64".format(period)
    print(formated_full)
