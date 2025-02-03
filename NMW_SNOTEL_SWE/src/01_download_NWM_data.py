import os
import pandas as pd
import numpy as np
import xarray as xr
import geopandas as gpd

import nwm
import param_nwm3
import misc

# Read NWM data
precip = nwm.s3_nwm3_zarr_bucket('CONUS/zarr/forcing/precip.zarr')['RAINRATE']*3600 # mm/s --> mm/hr
t2d = nwm.s3_nwm3_zarr_bucket('CONUS/zarr/forcing/t2d.zarr')['T2D']-273.15 # K --> C
sneqv = nwm.s3_nwm3_zarr_bucket('CONUS/zarr/ldasout.zarr')['SNEQV'] # kg/m^2 --> mm
snowh = nwm.s3_nwm3_zarr_bucket('CONUS/zarr/ldasout.zarr')['SNOWH']*1000 # m --> mm

# Read SNOTEL Site Info
gdf_snotel_site_info_utm12n_nad83 = gpd.read_file('../out/00_download_SNOTEL_data/info/snotel_site_info_utm12n_nad83.parquet.gzip')
gdf_snotel_site_info_nwm_lcc = gdf_snotel_site_info_utm12n_nad83.to_crs(param_nwm3.crs_nwm_proj4_lcc)


savedir = os.path.join(param_nwm3.out_dir,os.path.basename(__file__).split('.')[0])
misc.makedir(savedir)


# SNEQV
dfs = []
for i in gdf_snotel_site_info_nwm_lcc.index:
    site_code = gdf_snotel_site_info_nwm_lcc.loc[i, 'code']
    site_name = gdf_snotel_site_info_nwm_lcc.loc[i, 'name']
    print('SNEQV:',site_code)
    site_x = gdf_snotel_site_info_nwm_lcc.loc[i, 'geometry'].x
    site_y = gdf_snotel_site_info_nwm_lcc.loc[i, 'geometry'].y

    df = sneqv.sel(x=site_x, y=site_y, method='nearest').to_dataframe()['SNEQV']
    df = df.rename(site_code)
    df.index.name = 'datetime'

    dfs.append(df)

dfs = pd.concat(dfs, axis=1)
dfs.to_parquet(os.path.join(savedir,'SNEQV_3H.parquet.gzip'),compression='gzip')
dfs.resample('D').mean().to_parquet(os.path.join(savedir,'SNEQV_D.parquet.gzip'),compression='gzip')

# SNOWH
dfs = []
for i in gdf_snotel_site_info_nwm_lcc.index:
    site_code = gdf_snotel_site_info_nwm_lcc.loc[i, 'code']
    site_name = gdf_snotel_site_info_nwm_lcc.loc[i, 'name']
    print('SNOWH:',site_code)
    site_x = gdf_snotel_site_info_nwm_lcc.loc[i, 'geometry'].x
    site_y = gdf_snotel_site_info_nwm_lcc.loc[i, 'geometry'].y

    df = snowh.sel(x=site_x, y=site_y, method='nearest').to_dataframe()['SNOWH']
    df = df.rename(site_code)
    df.index.name = 'datetime'

    dfs.append(df)

dfs = pd.concat(dfs, axis=1)
dfs.to_parquet(os.path.join(savedir,'SNOWH_3H.parquet.gzip'),compression='gzip')
dfs.resample('D').mean().to_parquet(os.path.join(savedir,'SNOWH_D.parquet.gzip'),compression='gzip')

# PRECIP
dfs = []
for i in gdf_snotel_site_info_nwm_lcc.index:
    site_code = gdf_snotel_site_info_nwm_lcc.loc[i, 'code']
    site_name = gdf_snotel_site_info_nwm_lcc.loc[i, 'name']
    print('PRECIP:',site_code)
    site_x = gdf_snotel_site_info_nwm_lcc.loc[i, 'geometry'].x
    site_y = gdf_snotel_site_info_nwm_lcc.loc[i, 'geometry'].y

    df = precip.sel(x=site_x, y=site_y, method='nearest').to_dataframe()['RAINRATE']
    df = df.rename(site_code)
    df.index.name = 'datetime'

    dfs.append(df)

dfs = pd.concat(dfs, axis=1)
dfs.to_parquet(os.path.join(savedir,'PRECIP_H.parquet.gzip'),compression='gzip')
dfs.resample('D').sum().to_parquet(os.path.join(savedir,'PRECIP_D.parquet.gzip'),compression='gzip')

# T2D
dfs = []
for i in gdf_snotel_site_info_nwm_lcc.index:
    site_code = gdf_snotel_site_info_nwm_lcc.loc[i, 'code']
    site_name = gdf_snotel_site_info_nwm_lcc.loc[i, 'name']
    print('T2D:',site_code)
    site_x = gdf_snotel_site_info_nwm_lcc.loc[i, 'geometry'].x
    site_y = gdf_snotel_site_info_nwm_lcc.loc[i, 'geometry'].y

    df = t2d.sel(x=site_x, y=site_y, method='nearest').to_dataframe()['T2D']
    df = df.rename(site_code)
    df.index.name = 'datetime'

    dfs.append(df)

dfs = pd.concat(dfs, axis=1)
dfs.to_parquet(os.path.join(savedir,'T2D_H.parquet.gzip'),compression='gzip')
dfs.resample('D').mean().to_parquet(os.path.join(savedir,'T2D_D.parquet.gzip'),compression='gzip')
