# copy the utils functionalities into cupy objects
from datetime import datetime

import cupy as cp
import numpy as np
import pandas

from . import utils


__critvals = cp.array(utils.__critvals)
__critval_h = cp.array(utils.__critval_h)
__critval_period = cp.array(utils.__critval_period)
__critval_level = cp.array(utils.__critval_level)
__critval_mr = utils.__critval_mr.tolist()

check = utils.check

def get_critval(h, period, level, mr):
    
    # Sanity check
    check(h, period, level, mr)

    index = cp.zeros(4, dtype=cp.int)

    # Get index into table from arguments
    index[0] = next(i for i, v in enumerate(__critval_mr) if v == mr)
    index[1] = cp.where(level == __critval_level)[0][0]
    index[2] = (cp.abs(__critval_period - period)).argmin()
    index[3] = cp.where(h == __critval_h)[0][0]
    
    # For legacy reasons, the critvals are scaled by sqrt(2)
    return __critvals[tuple(index)] * cp.sqrt(2)

_find_index_date = utils._find_index_date

def crop_data_dates(data, dates, start, end):
    """ Crops the input data and the associated
    dates w.r.t. the provided start and end
    datetime object.

    Parameters
    ----------
    data: ndarray of shape (N, W, H)
        Here, N is the number of time
        series points per pixel and W
        and H are the width and the height
        of the image, respectively.
    dates : list of datetime objects
        Specifies the dates of the elements
        in data indexed by the first axis
        n_chunks : int or None, default None
    start : datetime
        The start datetime object
    end : datetime
        The end datetime object

    Returns
    -------
    Returns: data, dates
        The cropped data array and the
        cropped list. Only those images
        and dates that are with the start/end
        period are contained in the returned
        objects.
    """

    start_idx = _find_index_date(dates, start)
    end_idx = _find_index_date(dates, end)

    data_cropped = data[start_idx:end_idx, :, :]
    dates_cropped = list(cp.array(dates)[start_idx:end_idx])

    return data_cropped, dates_cropped


def compute_lam(N, hfrac, level, period):
    
    check(hfrac, period, 1 - level, "max")

    return get_critval(hfrac, period, 1 - level, "max")

compute_end_history = utils.compute_end_history

def map_indices(dates):
    
    return cp.array(utils.map_indices(dates))
