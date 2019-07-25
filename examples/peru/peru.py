import time
import copy
import numpy
numpy.warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import matplotlib
from datetime import datetime

from bfast import BFASTGPU
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

breaks = model_gpu.breaks
means = model_gpu.means

fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(10, 10))
no_breaks_indices = (breaks == -1)
means[no_breaks_indices] = 0
means[means > 0] = 0
im = axes.imshow(means, cmap='viridis', vmin=means.min(), vmax=None)
fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
fig.colorbar(im, cax=cbar_ax)

breaks_plot = breaks.astype(numpy.float)
breaks_plot[breaks == -2] = numpy.nan
breaks_plot[breaks == -1] = numpy.nan
breaks_plot[means >= 0] = numpy.nan

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

breaks_plot_years = copy.deepcopy(breaks_plot)
breaks_plot_years[breaks_plot <= idx_start_2011] = 0
breaks_plot_years[numpy.where(numpy.logical_and(idx_start_2011 < breaks_plot, breaks_plot <= idx_start_2012))] = 1
breaks_plot_years[numpy.where(numpy.logical_and(idx_start_2012 < breaks_plot, breaks_plot <= idx_start_2013))] = 2
breaks_plot_years[numpy.where(numpy.logical_and(idx_start_2013 < breaks_plot, breaks_plot <= idx_start_2014))] = 3
breaks_plot_years[numpy.where(numpy.logical_and(idx_start_2014 < breaks_plot, breaks_plot <= idx_start_2015))] = 4
breaks_plot_years[numpy.where(numpy.logical_and(idx_start_2015 < breaks_plot, breaks_plot <= idx_start_2016))] = 5
breaks_plot_years[numpy.where(idx_start_2016 < breaks_plot)] = 6

cmap = plt.get_cmap("gist_rainbow")
cmaplist = [cmap(i) for i in range(cmap.N)]
cmap = cmap.from_list('Custom cmap', cmaplist, cmap.N)

bounds = numpy.linspace(0, 6, 7)
norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)

fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(10, 10))
im = axes.imshow(breaks_plot_years, cmap=cmap, vmin=0, vmax=6, norm=norm)
fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
fig.colorbar(im, cax=cbar_ax, ticks=[0, 1, 2, 3, 4, 5, 6])
labels = cbar_ax.set_yticklabels(['2010', '2011', '2012', '2013', '2014', '2015', '2016'])

plt.show()