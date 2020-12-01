# REMEMBER TO RUN:
# $ futhark pkg add github.com/diku-dk/sorts
# $ futhark pkg sync

bfastfinal.py: bfastfinal.fut helpers.fut
	futhark pyopencl --library bfastfinal.fut
