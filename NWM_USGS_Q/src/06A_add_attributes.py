import os
import sys
sys.path.append('../src')

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


import matplotlib.pyplot as plt
import matplotlib

savedir = os.path.join(param_nwm3.out_dir,os.path.basename(__file__).split('.')[0])
savedir_parquet = os.path.join(savedir,'parquet')
savedir_csv = os.path.join(savedir,'csv')
savedir_giuseppe_share = os.path.join(savedir,'giuseppe_share')
misc.makedir(savedir_giuseppe_share)
misc.makedir(savedir_parquet)
misc.makedir(savedir_csv)

dir_calc_metrics = '../out/06_calc_metrics/parquet'
start_wy = 2003
end_wy = 2022

frequencies = ['hourly','daily']


for freq in frequencies:
    gp_site_info_metrics = os.path.join(dir_calc_metrics,'{}_{}_{}_utm12n_nad83.parquet.gzip'.format(freq,str(start_wy),str(end_wy)))
    gdf_site_info_metrics = gpd.read_parquet(gp_site_info_metrics)




    basin_climate_category = {
        'RDB':'M',
        'SPW':'M',
        'SC':'M',
        'RS':'M',
        'UG':'W',
        'SA':'W',
        'VE':'W',
        'LG':'W',
        'BW':'W',
        'LCO':'O',
        'CO':'O',
        'SJ':'O'
    }


    gdf_site_info_metrics['Climate'] = gdf_site_info_metrics['NAME_ABR'].map(basin_climate_category)
    gdf_site_info_metrics['ELE_m'] = gdf_site_info_metrics['alt_va']*0.3048 # Feet to Meters
    gdf_site_info_metrics = gdf_site_info_metrics.rename(columns={'site_no':'USGS_ID',
                                                                'NAME':'M_BASIN',
                                                                'NAME_ABR':'M_BASIN_ABR',
                                                                'huc_cd':'HUC8',
                                                                'drain_area_km2':'DAREA_km2'})
    gdf_site_info_metrics.to_parquet(os.path.join(savedir_parquet,gp_site_info_metrics.split('/')[-1]))
    gdf_site_info_metrics.to_csv(os.path.join(savedir_csv,gp_site_info_metrics.split('/')[-1].replace('.parquet.gzip','.csv')))



    # For Giuseppe
    gdf_site_info_metrics = gdf_site_info_metrics.rename(columns={'dec_lat_va':'LAT_d',
                                                        'dec_long_va':'LON_d',
                                                        })
    rearrange_cols = [
                    'USGS_ID',
                    'HUC8',
                    'M_BASIN',
                    'M_BASIN_ABR',
                    'Climate',
                    'LAT_d','LON_d','ELE_m',
                    'DAREA_km2'
                    ]
    rearrange_cols.reverse()
    for col in rearrange_cols:
        first_col = gdf_site_info_metrics.pop(col)
        gdf_site_info_metrics.insert(0, col, first_col)

    gdf_site_info_metrics.to_csv(os.path.join(savedir_giuseppe_share,gp_site_info_metrics.split('/')[-1].replace('.parquet.gzip','.csv')))
    gdf_site_info_metrics.loc[:,['USGS_ID','HUC8','M_BASIN','M_BASIN_ABR','Climate','LAT_d','LON_d','ELE_m','DAREA_km2','geometry']].to_file(os.path.join(savedir_giuseppe_share,gp_site_info_metrics.split('/')[-1].replace('.parquet.gzip','.shp')))

