.. _Getting Started:

Getting Started
===============

The following example shows how to apply BFASTMonitor on a medium-sized dataset.

.. literalinclude:: ../../examples/howto.py
    :start-after: # load packages
    :end-before: # define history and monitoring period and crop input data
    
First, a dataset is downloaded and stored in the 'data' directory (if the files do not exist yet). 
Afterwards, a Numpy array containing the satellite time series data is loaded as well as a text file that contains the
dates for the satellite images (i.e., a datetime index for the first dimension of 'data_orig')
    
.. literalinclude:: ../../examples/howto.py
    :start-after: # define history and monitoring period and crop input data
    :end-before: # apply BFASTMonitor using the OpenCL backend and the first device (e.g., GPU)
    
Next, the start of the history period as well as the start and end of the monitoring period are defined. Given these
datetimes, the data array and the dates are "cropped" .

.. literalinclude:: ../../examples/howto.py
    :start-after: # apply BFASTMonitor using the OpenCL backend and the first device (e.g., GPU) 
    
Finally, the BFASTMonitor model is defined and fitted using the 'opencl' backend. The data array is processed in 5 chunks.
