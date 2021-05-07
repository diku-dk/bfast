import os
import wget
import time
import copy
from functools import wraps
from datetime import datetime

import numpy as np
# np.warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import matplotlib

from bfast import BFASTMonitor
from bfast.monitor.utils import crop_data_dates


def cached(file_name):
    def outer_fun(func):
        @wraps(func)
        def inner_fun(*args, **kwargs):
            try:
                retval = np.load(file_name + ".npy")
            except:
                retval = func(*args, **kwargs)
                if retval is None:
                    raise ValueError("Function must return")
                np.save(file_name, retval)
            return retval
        return inner_fun
    return outer_fun

def run_bfast_(backend,
               k=3,
               freq=365,
               trend=False,
               hfrac=0.25,
               level=0.05,
               start_hist = datetime(2002, 1, 1),
               start_monitor = datetime(2010, 1, 1),
               end_monitor = datetime(2018, 1, 1)):

    print("Running the {} backend".format(backend))

    # download and parse input data
    ifile_meta = "data/peru_small/dates.txt"
    ifile_data = "data/peru_small/data.npy"

    if not os.path.isdir("data/peru_small"):
        os.makedirs("data/peru_small")

        if not os.path.exists(ifile_meta):
            url = 'https://sid.erda.dk/share_redirect/fcwjD77gUY/dates.txt'
            wget.download(url, ifile_meta)
        if not os.path.exists(ifile_data):
            url = 'https://sid.erda.dk/share_redirect/fcwjD77gUY/data.npy'
            wget.download(url, ifile_data)

    data_orig = np.load(ifile_data)
    with open(ifile_meta) as f:
        dates = f.read().split('\n')
        dates = [datetime.strptime(d, '%Y-%m-%d') for d in dates if len(d) > 0]

    data, dates = crop_data_dates(data_orig, dates, start_hist, end_monitor)

    # fit BFASTMontiro model
    model = BFASTMonitor(
            start_monitor,
            history="ROC",
            freq=freq,
            k=k,
            hfrac=hfrac,
            trend=trend,
            level=level,
            backend=backend,
            verbose=0,
            device_id=0,
            detailed_results=True,
    )

    start_time = time.time()
    if backend == "opencl":
        model.fit(data, dates, n_chunks=5, nan_value=-32768)
    else:
        model.fit(data, dates, nan_value=-32768)
    end_time = time.time()
    print("All computations have taken {} seconds.".format(end_time - start_time))

    # visualize results
    breaks = model.breaks
    means = model.means
    magnitudes = model.magnitudes
    valids = model.valids
    history_starts = model.history_starts
    return breaks, means, magnitudes, valids, history_starts


def compare(name, arr_p, arr_o):
    if arr_p.shape != arr_o.shape:
        raise ValueError("Array shapes must match")
    for i in range(arr_p.shape[0]):
        for j in range(arr_p.shape[1]):
            ap = arr_p[i, j]
            ao = arr_o[i, j]
            if not np.isclose(ap, ao):
                diff = abs(ap - ao)
                print("{} results at [{},{}] differ by {}".format(name, i, j, diff))
                exit()


# def run_bfast(backend):
#     @cached(backend)
#     def fun(backend):
#         return run_bfast_(backend)
#     return fun(backend)
def run_bfast(backend):
    return run_bfast_(backend)

def run_bfast_cached(backend):
    @cached(backend)
    def fun(backend):
        return run_bfast_(backend)
    return fun(backend)


def test(quantifier, pred, actual, expect, rel_err=False):
    stmt = pred(actual, expect)
    check = quantifier(stmt)
    if check:
      print("\033[92m PASSED \033[0m")
    else:
      print("\033[91m FAILED \033[0m")
      inds = np.where(~stmt)
      print("| Num. differences", np.sum(~stmt))
      print("| Expected", expect[inds])
      print("| Actual  ", actual[inds])
      if rel_err:
        print("| Relative absolute error")
        rel_err = np.abs((expect - actual)/expect)
        rel_err = rel_err[~np.isnan(rel_err)]
        per_err = rel_err * 100
        print("| Max error  {:10.5e} ({:.4f}%)".format(np.max(rel_err),
                                                       np.max(per_err)))
        print("| Min error  {:10.5e} ({:.4f}%)".format(np.min(rel_err),
                                                       np.min(per_err)))
        print("| Mean error {:10.5e} ({:.4f}%)".format(np.mean(rel_err),
                                                       np.mean(per_err)))


if __name__ == "__main__":
    # breaks_p, means_p, magnitudes_p, valids_p = run_bfast("python")
    breaks_p, means_p, magnitudes_p, valids_p, hist_p = run_bfast_cached("python-mp")
    breaks_o, means_o, magnitudes_o, valids_o, hist_o = run_bfast("opencl")
    # compare("breaks", breaks_p, breaks_o)
    # compare("means", means_p, means_o)

    breaks_diff = np.abs(breaks_p - breaks_o)
    means_diff = np.abs(means_p - means_o)
    magnitudes_diff = np.abs(magnitudes_p - magnitudes_o)
    valids_diff = np.abs(valids_p - valids_o)

    plt.imshow(breaks_p, cmap="Greys")
    plt.savefig("breaks_py.png")

    plt.clf()
    plt.imshow(breaks_o, cmap="Greys")
    plt.savefig("breaks_op.png")

    plt.clf()


    plt.imshow(means_p, cmap="Greys")
    plt.savefig("means_py.png")

    plt.clf()
    plt.imshow(means_o, cmap="Greys")
    plt.savefig("means_op.png")

    plt.clf()


    print("opencl_breaks", np.min(breaks_o), np.max(breaks_o))
    print("python_breaks", np.min(breaks_p), np.max(breaks_p))

    print("opencl_means", np.min(means_o), np.max(means_o))
    print("python_means", np.min(means_p), np.max(means_p))

    print("opencl_magnitudes", np.min(magnitudes_o), np.max(magnitudes_o))
    print("python_magnitudes", np.min(magnitudes_p), np.max(magnitudes_p))

    print("opencl_valids", np.min(valids_o), np.max(valids_o))
    print("python_valids", np.min(valids_p), np.max(valids_p))

    print("diff_means", np.min(means_diff), np.max(means_diff))
    print("diff_breaks", np.min(breaks_diff), np.max(breaks_diff))
    print("diff_magnitudes", np.min(magnitudes_diff), np.max(magnitudes_diff))
    print("diff_valids", np.min(valids_diff), np.max(valids_diff))

    plt.imshow(breaks_diff, cmap="Greys")
    plt.savefig("breaks_diff.png")
    plt.clf()

    plt.imshow(means_diff, cmap="Greys")
    plt.savefig("means_diff.png")
    plt.clf()

    plt.imshow(magnitudes_diff, cmap="Greys")
    plt.savefig("magnitudes_diff.png")

    plt.clf()
    plt.imshow(valids_diff, cmap="Greys")
    plt.savefig("valids_diff.png")

    print("np.all(breaks_o == breaks_p):", end="")
    test(np.all, np.equal, breaks_o, breaks_p)
    print("np.all(hist_o == hist_p):", end="")
    test(np.all, np.equal, hist_o, hist_p)
    print("np.all(np.isclose(means_o, means_p)):", end="")
    test(np.all, np.isclose, means_o, means_p, rel_err=True)
    print("np.all(np.isclose(magnitudes_o, magnitudes_p)):", end="")
    test(np.all, np.isclose, magnitudes_o, magnitudes_p, rel_err=True)
    print("np.all(valids_o == valids_p):", end="")
    test(np.all, np.equal, valids_o, valids_p)
