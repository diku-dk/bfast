#!/bin/bash
FUTHARK_INCREMENTAL_FLATTENING=1 futhark pyopencl --library bfastfinaldetailed.fut
FUTHARK_INCREMENTAL_FLATTENING=1 futhark pyopencl --library bfastfinal.fut
futhark pyopencl --library bfastdistrib.fut
futhark pyopencl --library bfastdistribdetailed.fut
