import os

import xarray as xr
import numpy as np
import pandas as pd
import geopandas as gpd
import datetime

import ulmo

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

import param_nwm3
import misc
import snotel


# Making output directories
savedir = os.path.join(param_nwm3.out_dir,os.path.basename(__file__).split('.')[0])
misc.makedir(savedir)

start_date = '1978-01-01'
end_date = datetime.date.today()

# Reading AZ-HUC8 boundary
gdf_wbdhu8_basins_az_utm12n = gpd.read_file(param_nwm3.gp_wbdhu8_basins_az_utm12n_nad83)
gdf_wbdhu8_boundary_az_utm12n = gpd.read_file(param_nwm3.gp_wbdhu8_boundary_az_utm12n_nad83)

# Get Snotel Site Information
gdf_site_info = snotel.get_snotel_info()
gdf_site_info_utm12n_nad83 = gdf_site_info.to_crs(param_nwm3.crs_utm12n_nad83)
gdf_site_info_az_utm12n_nad83 = gpd.clip(gdf_site_info_utm12n_nad83,gdf_wbdhu8_boundary_az_utm12n) # Clip Snotel Sites to AOI


# Save Site Info
savedir_snotel_data_site_info = os.path.join(savedir,'info')
misc.makedir(os.path.join(savedir_snotel_data_site_info))
gdf_site_info_az_utm12n_nad83.to_parquet(os.path.join(savedir_snotel_data_site_info,'snotel_site_info.parquet.gzip'),compression='gzip')
gdf_site_info_az_utm12n_nad83.to_file(os.path.join(savedir_snotel_data_site_info,'snotel_site_info.shp'))
gdf_site_info_az_utm12n_nad83.to_csv(os.path.join(savedir_snotel_data_site_info,'snotel_site_info.csv'))


sel_vars = {'D':['WTEQ','SNWD','PRCPSA','TOBS','TMIN','TMAX','TAVG'],
            'H':['WTEQ','SNWD','PRCPSA','TOBS','TMIN','TMAX','TAVG']}

# sel_vars = {'H':['SNWD']}


savedir_snotel_data_vars = os.path.join(savedir,'parquet')
misc.makedir(os.path.join(savedir_snotel_data_vars))

for temporal_resolution in list(sel_vars.keys()):                           # Temporal Resolution
    for variablecode in sel_vars[temporal_resolution]:                # Variable
        variablecode = 'SNOTEL:'+variablecode+'_'+temporal_resolution

        dfs_snotel = []
        for sitecode in gdf_site_info_az_utm12n_nad83.index:     # Site
            # Get Snotel Data
            
            dict_snotel = snotel.get_snotel_data(sitecode, variablecode=variablecode, start_date=start_date,end_date=end_date)
            if dict_snotel != None:
                df_snotel = dict_snotel['values'][['value']]
                df_snotel.columns = [sitecode.split(':')[-1]]
                dfs_snotel.append(df_snotel)
        dfs_snotel = pd.concat(dfs_snotel,axis=1)
        dfs_snotel.to_parquet(os.path.join(savedir_snotel_data_vars,variablecode.split(':')[-1]+'.parquet.gzip'),compression='gzip')




