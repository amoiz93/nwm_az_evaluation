import os

import xarray as xr
import numpy as np
import pandas as pd

import geopandas as gpd
import pyogrio
import fiona
import pyproj

import dataretrieval.nwis as nwis

import usgs
import nid
import nwm
import param_nwm3
import misc


# Making output directories
savedir = os.path.join(param_nwm3.out_dir,os.path.basename(__file__).split('.')[0])
savedir_parquet = os.path.join(savedir,'nwm3','parquet')
for dir in [savedir_parquet]:
    misc.makedir(dir)

gdf_site_info = gpd.read_parquet(os.path.join('../out/02_check_flags_USGS_Q_data','usgs_q_info_flagged_utm12n_wgs84.parquet.gzip'))
ds_chrtout = nwm.s3_nwm3_zarr_bucket('CONUS/zarr/chrtout.zarr')

start_date = '1979-01-01 00:00:00'
end_date = '2023-12-31 23:00:00'


gdf_site_info = gdf_site_info[gdf_site_info['nwm3g_comid'].notnull()|
                              gdf_site_info['nwm3p_comid'].notnull()]

for i in gdf_site_info.index:
    site_id = gdf_site_info.at[i,'site_no']
    nwm3g_comid = gdf_site_info.at[i,'nwm3g_comid']
    nwm3p_comid = gdf_site_info.at[i,'nwm3p_comid']
    if np.isnan(nwm3g_comid):
        comid = int(nwm3p_comid)
    else:
        comid = int(nwm3g_comid)


    print(i,site_id)
    df_discharge = ds_chrtout['streamflow'].sel(feature_id=comid).to_dataframe()['streamflow']
    df_discharge = df_discharge.rename(comid)

    df_discharge_hourly = pd.DataFrame(index=pd.date_range(start_date,end_date,freq='H'),columns=[comid])

    # Hourly
    misc.makedir(os.path.join(savedir_parquet,'hourly'))
    df_discharge_hourly.loc[:,comid] = df_discharge[df_discharge.index[0]:df_discharge.index[-1]]
    df_discharge_hourly.to_parquet(os.path.join(savedir_parquet,'hourly','{}.parquet.gzip'.format(site_id)),compression='gzip')

    # Daily
    misc.makedir(os.path.join(savedir_parquet,'daily'))
    df_discharge_daily = df_discharge_hourly.resample('D').mean()
    df_discharge_daily.to_parquet(os.path.join(savedir_parquet,'daily','{}.parquet.gzip'.format(site_id)),compression='gzip')

    # Monthly
    misc.makedir(os.path.join(savedir_parquet,'monthly'))
    df_discharge_monthly = df_discharge_hourly.resample('M').mean()
    df_discharge_monthly.to_parquet(os.path.join(savedir_parquet,'monthly','{}.parquet.gzip'.format(site_id)),compression='gzip')

    # Yearly
    misc.makedir(os.path.join(savedir_parquet,'yearly'))
    df_discharge_yearly = df_discharge_hourly.resample('Y').mean()
    df_discharge_yearly.to_parquet(os.path.join(savedir_parquet,'yearly','{}.parquet.gzip'.format(site_id)),compression='gzip')
    


