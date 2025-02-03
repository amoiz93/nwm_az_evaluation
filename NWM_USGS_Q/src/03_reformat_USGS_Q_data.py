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

'''
Flags are added in this script:
01 - exists [0=No Data Exists, 1=Data Exists]
02 - std_col [0=Non-Standard Columns, 1=Standard Columns]
03 - nldi_exists [0=No NLDI Basin, 1=NLDI Basin Exists]
04 - nid_ndams [Number of NID Dams Upstream]
05 - nid_maxstorage [Total Storage of NID Dams Upstream] [m3]
06 - nid_ID [NID ID of Dams Upstream]
07 - yuma [0=Not Yuma, 1=Yuma]
08 - hcdn2009 [0=Not HCDN2009, 1=HCDN2009]
09 - gagesii [0=Not GagesII, 1=GagesII]
10 - gagesii_class [Class of GagesII]
11 - nwm3g [0=No NWM3 Gage, 1=NWM3 Gage]
12 - nwm3g_comid [COMID of NWM3 at Gage location] # Used when NWM3 Gage is not available
13 - nwm21g [0=No NWM21 Gage, 1=NWM21 Gage]
14 - nwm21g_comid [COMID of NWM21 at Gage location]
15 - nwm3p_comid [COMID of NWM3 Catchment]
16 - nwm21p_comid [COMID of NWM21 Catchment]   # Used when NWM21 Gage is not available

# New flags Added
17 - inst_const [0=Not Constant, 1=Constant]
18 - daily_const [0=Not Constant, 1=Constant]
19 - daily_inst [0=Not Available, 1=Available] # Daily values derived from Instanteous Values
20 - daily_anom [0=Not Available, 1=Available] # Daily values derived from Instanteous Values (anomalous cases, manually identified)
'''


# Making output directories
savedir = os.path.join(param_nwm3.out_dir,os.path.basename(__file__).split('.')[0])
savedir_info = os.path.join(savedir,'usgs','info')
savedir_pkl = os.path.join(savedir,'usgs','pkl')
savedir_csv = os.path.join(savedir,'usgs','csv')
savedir_parquet = os.path.join(savedir,'usgs','parquet')
for dir in [savedir_info,savedir_pkl,savedir_csv,savedir_parquet]:
    misc.makedir(dir)

usgs_data_dir = '../out/01_std_col_USGS_Q_data'

gdf_site_info = gpd.read_parquet(os.path.join('../out/02_check_flags_USGS_Q_data','usgs_q_info_flagged_utm12n_wgs84.parquet.gzip'))

start_date = '1979-01-01 00:00:00'
end_date = '2023-12-31 23:00:00'

anomalous_sites = [
    '09473000',
    '09517490',
    '09512162',       # These are sites which have data at inst freq but nodata/incomplete data at daily freq
    '09408195'
    ]

frequencies = [
               'inst',
               'daily'
             ]

gdf_site_info['daily_inst'] = np.nan # Daily values derived from Instanteous Values
gdf_site_info['daily_anom'] = np.nan # Daily values derived from Instanteous Values (anomalous cases, manually identified)


for freq in frequencies:
    if freq == 'inst':
        gdf_site_info_freq = gdf_site_info[gdf_site_info['inst']==1]
    elif freq == 'daily':
        gdf_site_info_freq = gdf_site_info[(gdf_site_info['inst']==1)|
                                           (gdf_site_info['daily']==1)]
    
    if freq == 'inst':
        df_discharge_freq = pd.DataFrame(index=pd.date_range(start_date,end_date,freq='15min'),columns=['Q_cms'])
    elif freq == 'daily':
        df_discharge_freq = pd.DataFrame(index=pd.date_range(start_date,end_date,freq='D'),columns=['Q_cms'])

    for i in gdf_site_info_freq.index:
        site_id = gdf_site_info_freq.at[i,'site_no']
        print(freq,i,site_id)

        if freq == 'daily':
            if (gdf_site_info.at[i,'daily']==0): # Instantaneous exists but daily does not exist
                df_discharge = pd.read_parquet(os.path.join(usgs_data_dir,'inst','parquet','{}.parquet.gzip'.format(site_id)))
                gdf_site_info.at[i,'daily'] = 1 # Change Flag to Available
                gdf_site_info.at[i,'daily_inst'] = 1 # Change Flag to Available
            elif site_id in anomalous_sites:
                df_discharge = pd.read_parquet(os.path.join(usgs_data_dir,'inst','parquet','{}.parquet.gzip'.format(site_id)))
                gdf_site_info.at[i,'daily_anom'] = 1 # Change Flag to Available
            else:
                df_discharge = pd.read_parquet(os.path.join(usgs_data_dir,freq,'parquet','{}.parquet.gzip'.format(site_id)))
        else:
            df_discharge = pd.read_parquet(os.path.join(usgs_data_dir,freq,'parquet','{}.parquet.gzip'.format(site_id)))

        df_discharge = df_discharge[df_discharge['Q_flag'].notnull()] # Remove Null Values
        df_discharge = df_discharge[df_discharge['Q_cms']>=0] # Remove Negative Values
        df_discharge = df_discharge[df_discharge['Q_flag'].str.contains('A')] # Published values
        df_discharge = df_discharge.tz_localize(None)
        df_discharge = df_discharge['Q_cms']
        df_discharge_freq.columns = [site_id]
        df_discharge_freq.loc[:,site_id] = df_discharge[df_discharge_freq.index[0]:df_discharge_freq.index[-1]]


        # Anomymous Cases
        if freq == 'daily':
            if (gdf_site_info.at[i,'daily_inst'] == 1) | (gdf_site_info.at[i,'daily_anom'] == 1):
                df_discharge = df_discharge.resample('D').mean()

        # Check if all values are constant
        if freq == 'inst':
            if (df_discharge_freq[site_id].nunique() <= 1) == True:
                gdf_site_info.at[i,'inst_const'] = 1
            else:
                gdf_site_info.at[i,'inst_const'] = 0
        elif freq == 'daily':
            if (df_discharge_freq[site_id].nunique() <= 1) == True:
                gdf_site_info.at[i,'daily_const'] = 1
            else:
                gdf_site_info.at[i,'daily_const'] = 0            

        if freq == 'inst':
            #Intantenous (15-min)
            misc.makedir(os.path.join(savedir_parquet,'inst'))
            df_discharge_freq.to_parquet(os.path.join(savedir_parquet,'inst','{}.parquet.gzip'.format(site_id)),compression='gzip')
            
            # Hourly
            df_discharge_hourly = df_discharge_freq.resample('H').mean()
            misc.makedir(os.path.join(savedir_parquet,'hourly'))
            df_discharge_hourly.to_parquet(os.path.join(savedir_parquet,'hourly','{}.parquet.gzip'.format(site_id)),compression='gzip')

        elif freq == 'daily':
            # Daily
            misc.makedir(os.path.join(savedir_parquet,'daily'))
            df_discharge_freq.to_parquet(os.path.join(savedir_parquet,'daily','{}.parquet.gzip'.format(site_id)),compression='gzip')

            # Monthly
            misc.makedir(os.path.join(savedir_parquet,'monthly'))
            df_discharge_monthly = df_discharge_freq.resample('M').mean()
            df_discharge_monthly.to_parquet(os.path.join(savedir_parquet,'monthly','{}.parquet.gzip'.format(site_id)),compression='gzip')     

            # Yearly
            misc.makedir(os.path.join(savedir_parquet,'yearly'))
            df_discharge_yearly = df_discharge_freq.resample('Y').mean()
            df_discharge_yearly.to_parquet(os.path.join(savedir_parquet,'yearly','{}.parquet.gzip'.format(site_id)),compression='gzip')     


gdf_site_info.to_csv(os.path.join(savedir_info,'usgs_q_info_flagged_utm12n_wgs84.csv'))
gdf_site_info.to_parquet(os.path.join(savedir_info,'usgs_q_info_flagged_utm12n_wgs84.parquet.gzip'),compression='gzip')


