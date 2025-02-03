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


# Make all columns standard in the dataset


# Making output directories
savedir = os.path.join(param_nwm3.out_dir,os.path.basename(__file__).split('.')[0])



usgs_raw_data_dir = '../out/00_download_USGS_Q_data'


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

for freq in frequencies:
    
    # Read Info File
    gdf_site_info_freq = gpd.read_file(os.path.join(usgs_raw_data_dir,freq,'info','usgs_q_info_utm12n_nad83.parquet.gzip'))

    # Create Directories
    savedir_freq_info = os.path.join(savedir,freq,'info')
    savedir_freq_data_csv = os.path.join(savedir,freq,'csv')
    savedir_freq_data_parquet = os.path.join(savedir,freq,'parquet')
    for dir in [savedir_freq_info,savedir_freq_data_csv,savedir_freq_data_parquet]:
        misc.makedir(dir)


    for i in gdf_site_info_freq.index:
        
        site_id = gdf_site_info_freq.at[i,'site_no']
        flag_exists = gdf_site_info_freq.at[i,'exists']
        flag_std_col = gdf_site_info_freq.at[i,'std_col']
        print(freq,i,site_id,flag_exists,flag_std_col)

        if flag_exists == 1:
            df_discharge = pd.read_parquet(os.path.join(usgs_raw_data_dir,freq,'parquet','{}.parquet.gzip'.format(site_id)))

            # Fix Columns Headers
            if flag_std_col == 0:
                if freq == 'inst':
                    df_discharge = usgs.check_headers_inst_discharge(df_discharge)
                elif freq == 'daily':
                    df_discharge = usgs.check_headers_daily_discharge(df_discharge)

            if freq == 'inst':
                df_discharge = df_discharge.rename(columns={'00060':'Q_cfs','00060_cd':'Q_flag'})
            elif freq == 'daily':
                df_discharge = df_discharge.rename(columns={'00060_Mean':'Q_cfs','00060_Mean_cd':'Q_flag'})
            
            # Replace -999999.0 values with NaN
            df_discharge = df_discharge.replace(-999999.0,np.nan)

            # Unit Conversion
            df_discharge['Q_cms'] = df_discharge['Q_cfs']*(0.3048**3) # ft3/s --> m3/s

            # Save Data
            # df_discharge.to_csv(os.path.join(savedir_freq_data_csv,site_id+'.csv')) # Save CSV
            df_discharge.to_parquet(os.path.join(savedir_freq_data_parquet,site_id+'.parquet.gzip'),compression='gzip') # Save Parquet
    
    gdf_site_info_freq.to_csv(os.path.join(savedir_freq_info,'usgs_q_info_utm12n_nad83.csv'))   # Only for Reference
    # gdf_site_info_freq.to_file(os.path.join(savedir_freq_info,'usgs_q_info_utm12n_nad83.gpkg')) # Mainly use this
    gdf_site_info_freq.to_parquet(os.path.join(savedir_freq_info,'usgs_q_info_utm12n_nad83.parquet.gzip'),compression='gzip')

