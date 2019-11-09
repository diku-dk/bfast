# load packages
import os
import wget
import numpy
from datetime import datetime

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

data_orig = numpy.load(ifile_data)
with open(ifile_meta) as f:
    dates = f.read().split('\n')
    dates = [datetime.strptime(d, '%Y-%m-%d') for d in dates if len(d) > 0]

# define history and monitoring period and crop input data
from bfast.utils import crop_data_dates
start_hist = datetime(2002, 1, 1)
start_monitor = datetime(2010, 1, 1)
end_monitor = datetime(2018, 1, 1)
data, dates = crop_data_dates(data_orig, dates, start_hist, end_monitor)
print("First date: {}".format(dates[0]))
print("Last date: {}".format(dates[-1]))
print("Shape of data array: {}".format(data.shape))

# apply BFASTMonitor using the OpenCL backend and the first device (e.g., GPU)
from bfast import BFASTMonitor

model = BFASTMonitor(
            start_monitor,
            freq=365,
            k=3,
            hfrac=0.25,
            trend=False,
            level=0.05,
            backend='opencl',
            device_id=0,
        )
model.fit(data, dates, n_chunks=5, nan_value=-32768)

print("Detected breaks")
# -2 corresponds to not enough data for a pixel
# -1 corresponds to "no breaks detected"
# idx with isx>=0 corresponds to the position of the first break
print(model.breaks)
