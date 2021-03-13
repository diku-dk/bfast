import os
import argparse
import wget
import time
import copy
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from datetime import datetime
from functools import wraps

from bfast import BFASTMonitor
from bfast.monitor.utils import crop_data_dates

parser = argparse.ArgumentParser()
parser.add_argument("--backend", type=str, action="store", default="python",
                    help="backend for BFASTMonitor")
args = parser.parse_args()

# parameters
k = 3
freq = 365
trend = False
hfrac = 0.25
level = 0.05
start_hist = datetime(2002, 1, 1)
start_monitor = datetime(2010, 1, 1)
end_monitor = datetime(2018, 1, 1)

# download and parse input data
ifile_meta = "data/peru_small/dates.txt"
ifile_data = "data/peru_small/data.npy"

if not os.path.isdir("data/peru_small"):
    os.makedirs("data/peru_small")

if not os.path.exists(ifile_meta):
    url = "https://sid.erda.dk/share_redirect/fcwjD77gUY/dates.txt"
    wget.download(url, ifile_meta)
if not os.path.exists(ifile_data):
    url = "https://sid.erda.dk/share_redirect/fcwjD77gUY/data.npy"
    wget.download(url, ifile_data)

data_orig = np.load(ifile_data)
with open(ifile_meta) as f:
    dates = f.read().split("\n")
    dates = [datetime.strptime(d, "%Y-%m-%d") for d in dates if len(d) > 0]

data, dates = crop_data_dates(data_orig, dates, start_hist, end_monitor)
print("First date: {}".format(dates[0]))
print("Last date: {}".format(dates[-1]))
print("Shape of data array: {}".format(data.shape))

model = BFASTMonitor(
            start_monitor,
            freq=freq,
            k=k,
            history="ROC",
            hfrac=hfrac,
            trend=trend,
            level=level,
            backend=args.backend,
            verbose=0,
            device_id=0,
        )


FILENAME = "peru_hist.npy"

if os.path.exists(FILENAME):
  with open(FILENAME, "rb") as f:
    hist = np.load(f)
else:
  start_time = time.time()
  model.fit(data, dates, n_chunks=5, nan_value=-32768)
  end_time = time.time()
  print("All computations have taken {} seconds.".format(end_time - start_time))
  hist = model.history_starts
  with open(FILENAME, "wb") as f:
    np.save(f, hist)

  
# visualize stable history starts
not_enough_obs = hist == -2
hist = hist.astype(np.float)
hist[not_enough_obs] = np.nan

dates = np.array(dates)
idx_start_2003 = np.argmax((dates >= datetime(2003, 1, 1)) > False)
idx_start_2004 = np.argmax((dates >= datetime(2004, 1, 1)) > False)
idx_start_2005 = np.argmax((dates >= datetime(2005, 1, 1)) > False)
idx_start_2006 = np.argmax((dates >= datetime(2006, 1, 1)) > False)
idx_start_2007 = np.argmax((dates >= datetime(2007, 1, 1)) > False)
idx_start_2008 = np.argmax((dates >= datetime(2008, 1, 1)) > False)
idx_start_2009 = np.argmax((dates >= datetime(2009, 1, 1)) > False)
idx_start_2010 = np.argmax((dates >= datetime(2010, 1, 1)) > False)

hist_years = copy.deepcopy(hist)
hist_years[hist <= idx_start_2003] = 0
hist_years[np.where(np.logical_and(idx_start_2003 < hist, hist <= idx_start_2004))] = 1
hist_years[np.where(np.logical_and(idx_start_2004 < hist, hist <= idx_start_2005))] = 2
hist_years[np.where(np.logical_and(idx_start_2005 < hist, hist <= idx_start_2006))] = 3
hist_years[np.where(np.logical_and(idx_start_2006 < hist, hist <= idx_start_2007))] = 4
hist_years[np.where(np.logical_and(idx_start_2007 < hist, hist <= idx_start_2008))] = 5
hist_years[np.where(np.logical_and(idx_start_2008 < hist, hist <= idx_start_2009))] = 6
hist_years[np.where(np.logical_and(idx_start_2009 < hist, hist <= idx_start_2010))] = 7
hist_years[np.where(idx_start_2010 < hist)] = 8

bounds = np.linspace(0, 8, 9)
cmap = matplotlib.cm.get_cmap("gist_yarg", 9)

fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(10, 10))
im = axes.imshow(hist_years, cmap=cmap, vmin=0, vmax=8)
fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
fig.colorbar(im, cax=cbar_ax, ticks=[0, 1, 2, 3, 4, 5, 6, 7, 8])
labels = cbar_ax.set_yticklabels(["nan", "2003", "2004", "2005", "2006", "2007", "2008", "2009", "2010"])

plt.savefig("peru_roc.png")
