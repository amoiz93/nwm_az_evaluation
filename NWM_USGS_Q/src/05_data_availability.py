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
import eval_metrics

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
17 - inst_const [0=Not Constant, 1=Constant]
18 - daily_const [0=Not Constant, 1=Constant]
19 - daily_inst [0=Not Available, 1=Available] # Daily values derived from Instanteous Values
20 - daily_anom [0=Not Available, 1=Available] # Daily values derived from Instanteous Values (anomalous cases, manually identified)

# New flags Added
21 - DAyyyy [Daily Data Availability in yyyy]
22 - HAyyyy [Hourly Data Availability in yyyy]
'''

# Making output directories
savedir = os.path.join(param_nwm3.out_dir,os.path.basename(__file__).split('.')[0])
misc.makedir(savedir)

usgs_data_dir = '../out/03_reformat_USGS_Q_data/usgs/parquet/'
gdf_site_info = gpd.read_parquet(os.path.join('../out/03_reformat_USGS_Q_data/usgs/info','usgs_q_info_flagged_utm12n_wgs84.parquet.gzip'))

# Water Years
start_year = 1980
end_year = 2023
years_range = range(start_year,end_year+1)

# Daily Data Availability
gdf_site_info_daily = gdf_site_info[gdf_site_info['daily'] == 1]
for i in gdf_site_info_daily.index:
    site_id = gdf_site_info_daily.at[i,'site_no']
    df_usgs_obs = pd.read_parquet(os.path.join(usgs_data_dir,'daily','{}.parquet.gzip'.format(site_id)))
    df_usgs_obs = df_usgs_obs.iloc[:,0]
    df_usgs_obs = pd.DataFrame(df_usgs_obs)
    df_usgs_obs = usgs.add_water_year(df_usgs_obs)

    # Check Every Year
    for water_year in years_range:
        # Check Availability
        p_avail = usgs.check_percentage_availability_wy(df_usgs_obs,water_year)
        gdf_site_info.at[i,'DA'+str(water_year)] = p_avail
        print(i,site_id,water_year,p_avail)

# Hourly Data Availability
gdf_site_info_inst = gdf_site_info[gdf_site_info['inst'] == 1]
for i in gdf_site_info_inst.index:
    site_id = gdf_site_info_inst.at[i,'site_no']
    df_usgs_obs = pd.read_parquet(os.path.join(usgs_data_dir,'hourly','{}.parquet.gzip'.format(site_id)))
    df_usgs_obs = df_usgs_obs.iloc[:,0]
    df_usgs_obs = pd.DataFrame(df_usgs_obs)
    df_usgs_obs = usgs.add_water_year(df_usgs_obs)

    #df_usgs_obs = df_usgs_obs.resample('H').mean()

    # Check Every Year
    for water_year in years_range:
        # Check Availability
        p_avail = usgs.check_percentage_availability_wy(df_usgs_obs,water_year)
        gdf_site_info.at[i,'HA'+str(water_year)] = p_avail
        print(i,site_id,water_year,p_avail)


#gdf_site_info.to_file(os.path.join(savedir,'usgs_q_info_flagged.gpkg'),engine='pyogrio')
gdf_site_info = gdf_site_info.to_crs(param_nwm3.crs_utm12n_nad83)
gdf_site_info.to_csv(os.path.join(savedir,'usgs_q_info_flagged_utm12n_nad83.csv'))
gdf_site_info.to_parquet(os.path.join(savedir,'usgs_q_info_flagged_utm12n_nad83.parquet.gzip'),compression='gzip')

