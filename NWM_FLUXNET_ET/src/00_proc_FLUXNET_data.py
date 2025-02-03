import os
import glob
import pandas as pd
import geopandas as gpd
import xarray as xr
import numpy as np

import param_nwm3
import misc
import fluxnet

fluxnet_dir = os.path.join('../inp','Ameriflux-FLUXNET_FULLSET')
savedir = os.path.join(param_nwm3.out_dir,os.path.basename(__file__).split('.')[0])
savedir_info = os.path.join(savedir,'info')
savedir_parquet = os.path.join(savedir,'parquet')
misc.makedir(savedir_info)
misc.makedir(savedir_parquet)

gdf_wbdhu8_boundary_az_utm12n_nad83 = gpd.read_file(param_nwm3.gp_wbdhu8_boundary_az_utm12n_nad83)
gdf_wbdhu8_basins_az_utm12n_nad83 = gpd.read_file(param_nwm3.gp_wbdhu8_basins_az_utm12n_nad83)


# Process FLUXNET Site Information
df_fluxnet_site_info = pd.read_excel(os.path.join(fluxnet_dir,'AMF_AA-Flx_FLUXNET-BIF_CCBY4_20231211.xlsx'))
gdf_fluxnet_site_info_utm12n_nad83 = fluxnet.get_fluxnet_site_info(df_fluxnet_site_info).to_crs(param_nwm3.crs_utm12n_nad83)
gdf_fluxnet_site_info_az_utm12n_nad83 = gpd.clip(gdf_fluxnet_site_info_utm12n_nad83,gdf_wbdhu8_boundary_az_utm12n_nad83)
gdf_fluxnet_site_info_az_utm12n_nad83.to_parquet(os.path.join(savedir_info,'fluxnet_site_info_az_utm12n_nad83.parquet.gzip'),compression='gzip')

# Process FLUXNET Site Data
frequencies = ['monthly',
               'daily',
               'hourly']

vars = [
            'TA_F_MDS',    # degC
            'TA_F_MDS_QC', 
            'TA_ERA',
            'TA_F',
            'TA_F_QC', 
            'LE_F_MDS',    # W m-2
            'LE_F_MDS_QC',
            'LE_CORR',
            # 'LE_CORR_25',
            # 'LE_CORR_75',
            # 'LE_RANDUNC',
            # 'LE_CORR_JOINTUNC',
            # 'P',           # mm (missing in daily)
            'P_ERA',
            'P_F',         # mm
            'P_F_QC',
            'ET_F_MDS',    # mm
            'ET_CORR',     # mm
            ]

sites = list(gdf_fluxnet_site_info_az_utm12n_nad83.index)


for frequency in frequencies:
    for var in vars:
        print(frequency,var)
        df_fluxnet_site_data = fluxnet.get_FLXUNET_data(sites,var,frequency,fluxnet_dir,
                                    gdf_fluxnet_sites=gdf_fluxnet_site_info_az_utm12n_nad83)
        df_fluxnet_site_data.to_parquet(os.path.join(savedir_parquet,'fluxnet_'+var+'_'+frequency+'.parquet.gzip'),compression='gzip')

    
