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

# New flags Added
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
'''

# Make all columns standard in the dataset


# Making output directories
savedir = os.path.join(param_nwm3.out_dir,os.path.basename(__file__).split('.')[0])
misc.makedir(savedir)


usgs_data_dir = '../out/01_std_col_USGS_Q_data'

# Reading AZ-HUC8 boundary
gdf_wbdhu8_basins_az_lcc = gpd.read_parquet(param_nwm3.gp_wbdhu8_basins_az_lcc)
gdf_wbdhu8_boundary_az_lcc = gpd.read_parquet(param_nwm3.gp_wbdhu8_boundary_az_lcc)


# Read NWM Parameters
gdf_nwm3_reaches_az_lcc = gpd.read_parquet(param_nwm3.gp_nwm_reaches_az_lcc)
gdf_nwm3_catchments_az_lcc = gpd.read_parquet(param_nwm3.gp_nwm_catchments_az_lcc)
gdf_nwm3_waterbodies_az_lcc = gpd.read_parquet(param_nwm3.gp_nwm_waterbodies_az_lcc)
ds_nwm3_gages_info = nwm.get_chrtout_gages(nwm.s3_nwm3_zarr_bucket('CONUS/zarr/chrtout.zarr'))

gdf_nwm21_reaches_az_lcc = pyogrio.read_dataframe(os.path.join('/Users/amoiz2/Servers/phx/data/nwm/retrospective/az/const/nwm_params/nwm_reaches_az_lcc.shp'))
gdf_nwm21_catchments_az_lcc = pyogrio.read_dataframe(os.path.join('/Users/amoiz2/Servers/phx/data/nwm/retrospective/az/const/nwm_params/nwm_catchments_az_lcc.shp'))
gdf_nwm21_waterbodies_az_lcc = pyogrio.read_dataframe(os.path.join('/Users/amoiz2/Servers/phx/data/nwm/retrospective/az/const/nwm_params/nwm_waterbodies_az_lcc.shp'))
ds_nwm21_gages_info = nwm.get_chrtout_gages(nwm.s3_nwm21_zarr_bucket('chrtout.zarr'))

# Reading NID
gdf_nid = pyogrio.read_dataframe(nid.nid_national_files['gpkg_national_nid'])
nid_dam_completed_year = 2023 # Consider all dams constructed before this year

# Read USGS info
gdf_usgs_gages_ii_info = pyogrio.read_dataframe('../inp/usgs_gagesii_info/gagesII_9322_point_shapefile/gagesII_9322_sept30_2011.shp')
gdf_usgs_hcdn2009_info = gdf_usgs_gages_ii_info[gdf_usgs_gages_ii_info['HCDN_2009'] == 'yes']

# Read Manually Selected YUMA Sites
df_yuma_stations = pd.read_csv('../inp/yuma_excluded.csv')

# 01 - Merge Daily and Hourly Site Info
gdf_site_info_inst = gpd.read_parquet(os.path.join(usgs_data_dir,'inst','info','usgs_q_info_utm12n_nad83.parquet.gzip'))
gdf_site_info_daily = gpd.read_parquet(os.path.join(usgs_data_dir,'daily','info','usgs_q_info_utm12n_nad83.parquet.gzip'))
gdf_site_info = gdf_site_info_daily.copy()  
gdf_site_info = gdf_site_info.drop(columns=['exists','std_col'])

for i in gdf_site_info.index:
    
    site_id = gdf_site_info_inst.at[i,'site_no']
    flag_inst_exists = gdf_site_info_inst.at[i,'exists']
    flag_daily_exists = gdf_site_info_daily.at[i,'exists']
    print(i,site_id)

    # Combine Daily and Inst GDF
    gdf_site_info.at[i,'inst'] = flag_inst_exists
    gdf_site_info.at[i,'daily'] = flag_daily_exists

    if ((flag_inst_exists == 1) | (flag_daily_exists == 1)):
        # 02 - Check NID Storage

        # 02A - Check NLDI
        try:
            site_basin = nwm.get_basin_by_usgsgageid(site_id)
            #gdf_site_info.at[i,'nldi_poly'] = site_basin.iloc[0].geometry
            flag_nldi_exists = 1
        except:
            flag_nldi_exists = 0
        gdf_site_info.at[i,'nldi_exists'] = flag_nldi_exists

        # 02B - Check NID
        if flag_nldi_exists:
            gdf_upstream_reservoirs = gpd.clip(gdf_nid,site_basin.iloc[0].geometry)
            if (len(gdf_upstream_reservoirs) > 0):
                nid_year_completed = pd.to_numeric(gdf_upstream_reservoirs['yearCompleted'],errors='coerce')
                gdf_upstream_reservoirs = gdf_upstream_reservoirs[nid_year_completed<=nid_dam_completed_year]
                nid_upstream_ndams = len(gdf_upstream_reservoirs)
                nid_total_upstream_storage = gdf_upstream_reservoirs['maxStorage']
                nid_total_upstream_storage = pd.to_numeric(nid_total_upstream_storage,errors='coerce')
                nid_total_upstream_storage = nid_total_upstream_storage.sum()*1233.48 # Acre-Feet to m3
                nid_upstream_nidID = gdf_upstream_reservoirs['nidId'].to_list()
            else:
                nid_upstream_ndams = 0
                nid_total_upstream_storage = 0
                nid_upstream_nidID = None

            gdf_site_info.at[i,'nid_ndams'] = nid_upstream_ndams
            gdf_site_info.at[i,'nid_maxstorage'] = nid_total_upstream_storage
            gdf_site_info.at[i,'nid_ID'] = nid_upstream_nidID

        # 02C - Check Yuma Sites
        site_name = gdf_site_info.at[i,'station_nm']
        if 'YUMA' in site_name:
            gdf_site_info.at[i,'yuma'] = 1
        elif int(site_id) in df_yuma_stations['site_no'].to_list():
            gdf_site_info.at[i,'yuma'] = 1
        else:
            gdf_site_info.at[i,'yuma'] = 0

        # 03 - Check HCDN2009/GagesII
        # 03A - Check HCDN2009
        if (gdf_usgs_hcdn2009_info['STAID'] == site_id).any():
            gdf_site_info.at[i,'hcdn2009'] = 1
        else:
            gdf_site_info.at[i,'hcdn2009'] = 0

        # 03B - Check GagesII
        if (gdf_usgs_hcdn2009_info['STAID'] == site_id).any():
            gdf_site_info.at[i,'gagesii'] = 1
            gdf_site_info.at[i,'gagesii_class'] = (gdf_usgs_gages_ii_info[gdf_usgs_gages_ii_info['STAID']==site_id]['CLASS'].values[0])
        else:
            gdf_site_info.at[i,'gagesii'] = 0

        # 04A - Check Availability in NWM3
        # Get Gage Information (Availability & COMID) from NWM if it exists
        if (ds_nwm3_gages_info == site_id).any():
            gdf_site_info.at[i,'nwm3g'] = 1
            gdf_site_info.at[i,'nwm3g_comid'] = (ds_nwm3_gages_info[ds_nwm3_gages_info==site_id].index.values[0])
            if np.isnan(gdf_site_info.at[i,'nwm3g_comid']):
                gdf_site_info.at[i,'nwm3g'] = 0
        else:
            gdf_site_info.at[i,'nwm3g'] = 0

        # 04B - Check Availability in NWM21
        if (ds_nwm21_gages_info == site_id).any():
            gdf_site_info.at[i,'nwm21g'] = 1
            gdf_site_info.at[i,'nwm21g_comid'] = (ds_nwm21_gages_info[ds_nwm21_gages_info==site_id].index.values[0])
            if np.isnan(gdf_site_info.at[i,'nwm21g_comid']):
                gdf_site_info.at[i,'nwm21g'] = 0
        else:
            gdf_site_info.at[i,'nwm21g'] = 0


# 05A - Get COMID from NWM3 Catchments
gdf_site_info = gdf_site_info.sjoin(gdf_nwm3_catchments_az_lcc.to_crs(param_nwm3.crs_utm12n_nad83),how='left')
gdf_site_info = gdf_site_info.rename(columns={'ID':'nwm3p_comid'})
gdf_site_info = gdf_site_info.drop(columns=['Shape_Length','Shape_Area','index_right'])

# 05B - Get COMID from NWM21 Catchments
gdf_site_info = gdf_site_info.sjoin(gdf_nwm21_catchments_az_lcc.to_crs(param_nwm3.crs_utm12n_nad83),how='left')
gdf_site_info = gdf_site_info.rename(columns={'feature_id':'nwm21p_comid'})
gdf_site_info = gdf_site_info.drop(columns=['Shape_Leng','Shape_Area','index_right','source'])


gdf_site_info.to_csv(os.path.join(savedir,'usgs_q_info_flagged_utm12n_nad83.csv'))
# gdf_site_info.to_file(os.path.join(savedir,'usgs_q_info_flagged_utm12n_nad83.gpkg'),engine='pyogrio')
# gdf_site_info.to_file(os.path.join(savedir,'usgs_q_info_flagged_utm12n_nad83.shp'),engine='pyogrio')
gdf_site_info.to_parquet(os.path.join(savedir,'usgs_q_info_flagged_utm12n_nad83.parquet.gzip'),compression='gzip')
