import os
import sys
sys.path.append('../src')
import glob

import xarray as xr
import xvec
import rioxarray
import numpy as np
import pandas as pd
import geopandas as gpd
from rasterio.enums import Resampling
import dask

import misc
import param_nwm3

start_wy = 1981
end_wy = 2020


savedir = os.path.join('../out',os.path.basename(__file__).split('.')[0])
misc.makedir(savedir)

ds_crs = param_nwm3.crs_nwm_proj4_lcc
target_crs = param_nwm3.crs_utm12n_nad83

gdf_wbdhu8_boundary_az_lcc = gpd.read_file(param_nwm3.gp_wbdhu8_boundary_az_lcc)
gdf_wbdhu8_basins_az_lcc = gpd.read_file(param_nwm3.gp_wbdhu8_basins_az_lcc)
gdf_wbdhu8_basins_az_lcc.index = gdf_wbdhu8_basins_az_lcc['huc8']
gdf_wbdhu8_basins_az_utm12n_nad83 = gdf_wbdhu8_basins_az_lcc.to_crs(target_crs)

gdf_az_basins_utm12n = gpd.read_file(os.path.join('../../../data/az_nwm/const/basins/az_watersheds_simplified_utm12n_nad83.shp')).to_crs(target_crs)
gdf_az_basins_utm12n['geometry'] = gdf_az_basins_utm12n.simplify(1E03)
gdf_az_basins_lcc = gdf_az_basins_utm12n.to_crs(ds_crs)

ds_nwm_sneqv = xr.open_mfdataset(sorted(glob.glob('/Users/amoiz2/Servers/dropbox_asu/Personal/Work/20230807_ASU_SSEBE_Mascaro/Workspace/projects/20240903_NWM_ATUR_Deliverable_AY1/inp/nwm3/retrospective/ldasout/monthly/SNEQV/*.nc')))

# Create basin level SWE
ds_nwm_sneqv = ds_nwm_sneqv.rio.write_crs(ds_crs)
ds_nwm_sneqv_huc8 = ds_nwm_sneqv.xvec.zonal_stats(gdf_az_basins_lcc.geometry,x_coords='x',y_coords='y',stats='mean')
ds_nwm_sneqv_huc8_reproj = ds_nwm_sneqv_huc8.xvec.to_crs(geometry=target_crs) # reproject to UTM12N
ds_nwm_sneqv_huc8_reproj = ds_nwm_sneqv_huc8_reproj.sel(time=slice('2002-10-01','2022-09-30'))
ds_nwm_sneqv_huc8_reproj_LTM = ds_nwm_sneqv_huc8_reproj.groupby('time.month').mean('time')

gdf_az_basins_utm12n_final = gdf_az_basins_utm12n.copy()
month_strs = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
for month in range(1,13):
    print(month)
    month_str = month_strs[month-1]
    gdf_az_basins_utm12n_final[month_str] = ds_nwm_sneqv_huc8_reproj_LTM['SNEQV'].sel(month=month).values

# Final Results
gdf_az_basins_utm12n_analysis = gdf_az_basins_utm12n_final.copy()
gdf_az_basins_utm12n_analysis.index = gdf_az_basins_utm12n_analysis['NAME_ABR']
gdf_az_basins_utm12n_analysis = gdf_az_basins_utm12n_analysis.drop(columns=['NAME_ABR','Climate','area_m2','NAME','geometry'])
gdf_az_basins_utm12n_analysis = gdf_az_basins_utm12n_analysis.max(axis=1).sort_values(ascending=False)

# ds_nwm_sneqv_huc8_encoded = ds_nwm_sneqv_huc8_reproj.xvec.encode_cf()
# ds_stats_huc8_encoded.to_netcdf(os.path.join(savedir_nc,f'NWM_{start_wy}_{end_wy}_summary_VEC_HUC8.nc'),mode='w')




