import time
import numpy
numpy.warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import pandas
from datetime import datetime

from bfast import BFASTCPU, BFASTGPU
from bfast.utils import crop_data_dates

# input data
ifile_meta = "peru_ndmi_stack.grd"
ifile_data = "peru.npy"

# parameters
k = 3
freq = 365
trend = False
hfrac = 0.25
level = 0.05
start_hist = datetime(2002, 1, 1)
start_monitor = datetime(2010, 1, 1)
end_monitor = datetime(2018, 1, 1)

# pixel to be compared (CPU vs. GPU)
pos = (100, 100)

# load and crop data
data_orig = numpy.load(ifile_data)
with open(ifile_meta) as f:
    dates = f.read().split('\n')[25].split("time:")[1]
    dates = dates.split(":")
    dates = [datetime.strptime(d, '%Y-%m-%d') for d in dates]

data_orig, dates = crop_data_dates(data_orig, dates, start_hist, end_monitor)
print("First date: {}".format(dates[0]))
print("Last date: {}".format(dates[-1]))
print("Shape of data array: {}".format(data_orig.shape))

##############################################################################################
########################################### CPU ##############################################
##############################################################################################
# fit BFAST using the CPU implementation (single pixel)
model_cpu = BFASTCPU(
            start_monitor,
            freq=freq,
            k=k,
            hfrac=hfrac,
            trend=trend,
            level=level,
            )
data = data_orig.astype(numpy.float32)
data[data == -32768] = numpy.nan
model_cpu.fit(data[:, pos[0], pos[1]], dates)

dates_monitor = []

# collect dates for monitor period
for i in range(len(dates)):
    if start_monitor <= dates[i]:
        dates_monitor.append(dates[i])
dates_array = numpy.array(dates_monitor)
idx_start_2010 = numpy.argmax((dates_array >= datetime(2010, 1, 1)) > False)
idx_start_2011 = numpy.argmax((dates_array >= datetime(2011, 1, 1)) > False)
idx_start_2012 = numpy.argmax((dates_array >= datetime(2012, 1, 1)) > False)
idx_start_2013 = numpy.argmax((dates_array >= datetime(2013, 1, 1)) > False)
idx_start_2014 = numpy.argmax((dates_array >= datetime(2014, 1, 1)) > False)
idx_start_2015 = numpy.argmax((dates_array >= datetime(2015, 1, 1)) > False)
idx_start_2016 = numpy.argmax((dates_array >= datetime(2016, 1, 1)) > False)

# generate pandas time series
ts_y = pandas.Series(data[:, pos[0], pos[1]], dates)
ts_pred = pandas.Series(model_cpu.y_pred, dates)
ts_mosum = pandas.Series(model_cpu.mosum, dates_monitor)
ts_boundary_pos = pandas.Series(model_cpu.bounds, dates_monitor)
ts_boundary_neg = pandas.Series(-model_cpu.bounds, dates_monitor)

# generate first plot
f, axes = plt.subplots(2, sharex=True, figsize=(15, 10))
ts_y.plot(ax=axes[0], style="ro", ms=6, label="NDMI")
ts_pred.plot(ax=axes[0], style="g+", ms=6, label='predictions')
axes[0].set(ylabel='NDMI')

# generate second plot
ts_mosum.plot(ax=axes[1], style="b.", ms=6, label="MOSUM")
ts_boundary_neg.plot(ax=axes[1], style="k-", ms=6, label='')
ts_boundary_pos.plot(ax=axes[1], style="k-", ms=6, label='')
axes[1].set(ylabel='Deviation')
plt.axvline(x=dates[model_cpu.n + model_cpu.first_break], color='r', linestyle='--', label='first break')

plt.xlabel('Time')
f.legend()

##############################################################################################
########################################### GPU ##############################################
##############################################################################################
model_gpu = BFASTGPU(
            start_monitor,
            freq=freq,
            k=k,
            hfrac=hfrac,
            trend=trend,
            level=level,
            detailed_results=True,  # needed to plot
            verbose=1,
            device_id=0,
            )

start_time = time.time()
model_gpu.fit(data_orig, dates, n_chunks=5, nan_value=-32768)
end_time = time.time()
print("All computations have taken {} seconds.".format(end_time - start_time))

# plots
ts_y = pandas.Series(data[:, pos[0], pos[1]], dates)
ts_pred = pandas.Series(model_gpu.y_pred[:, pos[0], pos[1]], dates)
ts_mosum = pandas.Series(model_gpu.mosum[:, pos[0], pos[1]], dates_monitor)
ts_boundary_pos = pandas.Series(model_gpu.bounds, dates_monitor)
ts_boundary_neg = pandas.Series(-model_gpu.bounds, dates_monitor)

f, axes = plt.subplots(2, sharex=True, figsize=(15, 10))
ts_y.plot(ax=axes[0], style="ro", ms=6, label="NDMI")
ts_pred.plot(ax=axes[0], style="g+", ms=6, label='predictions')
axes[0].set(ylabel='NDMI')
ts_mosum.plot(ax=axes[1], style="b.", ms=6, label="MOSUM")
ts_boundary_neg.plot(ax=axes[1], style="k-", ms=6, label='')
ts_boundary_pos.plot(ax=axes[1], style="k-", ms=6, label='')
axes[1].set(ylabel='Deviation')
plt.axvline(x=dates[model_cpu.n + model_gpu.breaks[pos[0], pos[1]]], color='r', linestyle='--', label='first break')
plt.xlabel('Time')
f.legend()

plt.show()
