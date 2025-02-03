import os
import glob
import pandas as pd
import numpy as np
import geopandas as gpd
import xarray as xr

import param_nwm3
import misc

savedir = os.path.join(param_nwm3.out_dir,os.path.basename(__file__).split('.')[0])
misc.makedir(savedir)

gdf_site_info_utm12n_nad83 = gpd.read_file('../out/02_calc_metrics/daily_2003_2022_utm12n_nad83.parquet.gzip')
gdf_site_info_utm12n_nad83.to_file(os.path.join(savedir,'snotel_sites_utm12n_nad83.shp'))
gdf_site_info_utm12n_nad83.to_parquet(os.path.join(savedir,'snotel_sites_utm12n_nad83.parquet.gzip'),compression='gzip')

station_count = gdf_site_info_utm12n_nad83.groupby('M_BASIN_ABR').count()
station_count = station_count['code']

