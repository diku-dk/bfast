import numpy as np

import bfast
from datasets import *


Yt = harvest.reshape((harvest.shape[0], 1, 1))
bf = bfast.BFAST(frequency=harvest_freq,
                 h=0.15,
                 season_type="dummy",
                 max_iter=10,
                 max_breaks=None,
                 level=0.05,
                 verbose=1)

bf.fit(Yt=Yt, ti=harvest_dates)

number_trend_breakpoints = bf.n_trend_breakpoints[0,0]
trend_breakpoints = bf.trend_breakpoints
print(trend_breakpoints[:number_trend_breakpoints,0,0])

