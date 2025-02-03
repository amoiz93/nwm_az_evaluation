import os

import xarray as xr
import numpy as np
import pandas as pd
import geopandas as gpd

import dataretrieval.nwis as nwis

import usgs
import nid
import nwm
import param_nwm3
import misc

'''
Two Flags are added in this script:
01 - exists [0=No Data Exists, 1=Data Exists]
02 - std_col [0=Non-Standard Columns, 1=Standard Columns]
'''

# Making output directories
savedir = os.path.join(param_nwm3.out_dir,os.path.basename(__file__).split('.')[0])

# # Services
# # instantaneous values (iv)
# # daily values (dv)
# # statistics (stat)
# # site info (site)
# # discharge peaks (peaks)
# # discharge measurements (measurements)

frequencies = [
               'inst',
               'daily'
               ]

# Reading AZ-HUC8 boundary
gdf_wbdhu8_basins_az_lcc = gpd.read_file(param_nwm3.gp_wbdhu8_basins_az_lcc)
gdf_wbdhu8_boundary_az_lcc = gpd.read_file(param_nwm3.gp_wbdhu8_boundary_az_lcc)

# Get information of all sites
gdf_site_info = usgs.get_usgs_site_info_by_shapefile(states=['AZ','CA','NV','NM','CO','UT'],
                                       bounding_gdf=gdf_wbdhu8_boundary_az_lcc,
                                       from_crs=param_nwm3.crs_nad83,
                                       parameterCd='00060')

for freq in frequencies:
    # Create Flag Columns
    gdf_site_info_freq = gdf_site_info.copy()
    gdf_site_info_freq[['exists','std_col']] = 0 # Binary (0/1)
    

    # Create Directories
    savedir_freq_info = os.path.join(savedir,freq,'info')
    savedir_freq_data_csv = os.path.join(savedir,freq,'csv')
    savedir_freq_data_parquet = os.path.join(savedir,freq,'parquet')
    for dir in [savedir_freq_info,savedir_freq_data_csv,savedir_freq_data_parquet]:
        misc.makedir(dir)

    for i in gdf_site_info_freq.index:
        site_id = gdf_site_info_freq.at[i,'site_no']
        print(freq,i,site_id)

        # Download Discharge Data

        if freq == 'inst':
            df_discharge = usgs.download_usgs_inst_discharge(site_id)
        elif freq == 'daily':
            df_discharge = usgs.download_usgs_daily_discharge(site_id)
        

        # 01 - Check if Data Exists
        if df_discharge.empty == True:
            gdf_site_info_freq.at[i,'exists'] = 0
        else:
            gdf_site_info_freq.at[i,'exists'] = 1
            
            
            # Check Columns are Standard
            if freq == 'inst':
                std_col_headers = ['00060', '00060_cd', 'site_no']
            elif freq == 'daily':
                std_col_headers = ['00060_Mean', '00060_Mean_cd', 'site_no']

            if list(df_discharge.columns) == std_col_headers:
                gdf_site_info_freq.at[i,'std_col'] = 1
            else:
                gdf_site_info_freq.at[i,'std_col'] = 0

            # Save Data
            #df_discharge.to_csv(os.path.join(savedir_freq_data_csv,site_id+'.csv'))  # Save CSV
            df_discharge.to_parquet(os.path.join(savedir_freq_data_parquet,site_id+'.parquet.gzip'),compression='gzip') # Save Parquet
            
    gdf_site_info_freq.to_csv(os.path.join(savedir_freq_info,'usgs_q_info_utm12n_nad83.csv'))   # Only for Reference
    gdf_site_info_freq.to_crs(param_nwm3.crs_utm12n_nad83).to_parquet(os.path.join(savedir_freq_info,'usgs_q_info_utm12n_nad83.parquet.gzip'),compression='gzip') # Mainly use this
