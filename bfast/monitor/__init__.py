from .opencl import BFASTMonitorOpenCL
from .python import BFASTMonitorPython

# only import cupy related bfast if a gpu is available
try: 
    import cupy
    cupy.cuda.Device()
    gpu_available = True
except:
    gpu_available = False

if gpu_available:
    from .cupy import BFASTMonitorCuPy
