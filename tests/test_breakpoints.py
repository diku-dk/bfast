import os
import time
import copy
from datetime import datetime

import wget
import numpy as np
np.warnings.filterwarnings('ignore')
import numpy.testing as npt

from bfast import BFASTMonitor
from bfast.monitor.utils import crop_data_dates


def download_file(url, path):
    try:
        if not os.path.exists(path):
            wget.download(url, path)
    except:
        raise ConnectionError("Failed loading dataset {}".format(path))


def apply_bfastmonitor(k,
                       freq,
                       trend,
                       hfrac,
                       level,
                       start_hist,
                       start_monitor,
                       end_monitor,
                       backend,
                       meta_url,
                       data_url,
                       base_dir,
                       meta_dir,
                       data_dir
                       ):

    # download and parse input data
    download_file(meta_url, meta_dir)
    download_file(data_url, data_dir)

    data_orig = np.load(data_dir)

    with open(meta_dir) as f:
        dates = f.read().split('\n')
        dates = [datetime.strptime(d, '%Y-%m-%d') for d in dates if len(d) > 0]

    data, dates = crop_data_dates(data_orig, dates, start_hist, end_monitor)

    model = BFASTMonitor(
              start_monitor,
              freq=freq,
              k=k,
              hfrac=hfrac,
              trend=trend,
              level=level,
              backend=backend
            )

    model.fit(data, dates, n_chunks=5, nan_value=-32768)

    breaks = model.breaks
    return breaks


def peru(backend):
    return apply_bfastmonitor(k=3,
                              freq=365,
                              trend=False,
                              hfrac=0.25,
                              level=0.05,
                              start_hist=datetime(2002, 1, 1),
                              start_monitor=datetime(2010, 1, 1),
                              end_monitor=datetime(2018, 1, 1),
                              backend=backend,
                              meta_url="https://sid.erda.dk/share_redirect/EbHxrlTKSy",
                              data_url="https://sid.erda.dk/share_redirect/bQuufDuYqS",
                              base_dir="data/peru",
                              meta_dir="data/peru/peru.dates.txt",
                              data_dir="data/peru/peru.in.npy")


def test_peru_opencl():
    base_dir = "data/peru"
    if not os.path.isdir(base_dir):
        os.makedirs(base_dir)
    breaks_dir = "data/peru/peru.out.breaks.npy"
    breaks_url = "https://sid.erda.dk/share_redirect/BlAmRIImFV"
    download_file(breaks_url, breaks_dir)
    breaks_actual = peru("opencl")
    breaks_expected = np.load(breaks_dir)
    npt.assert_equal(breaks_expected, breaks_actual)
