import os
import pandas as pd
import numpy as np
import xarray as xr
import geopandas as gpd

import nwm
import param_nwm3
import misc

# Read NWM data
precip = nwm.s3_nwm3_zarr_bucket('CONUS/zarr/forcing/precip.zarr')['RAINRATE']*3600 # mm/s --> mm/hr
t2d = nwm.s3_nwm3_zarr_bucket('CONUS/zarr/forcing/t2d.zarr')['T2D']-273.15 # K --> C
sneqv = nwm.s3_nwm3_zarr_bucket('CONUS/zarr/ldasout.zarr')['SNEQV'] # kg/m^2 --> mm
snowh = nwm.s3_nwm3_zarr_bucket('CONUS/zarr/ldasout.zarr')['SNOWH']*1000 # m --> mm
