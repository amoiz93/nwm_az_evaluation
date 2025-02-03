import os
import pandas as pd
import numpy as np
import xarray as xr
import geopandas as gpd

import nwm
import param_nwm3
import misc

# def accet_to_et_df(df_accet):
#     df_et = df_accet.copy()
#     df_et[df_et<0] = 0
#     return df_et

# Read NWM data
precip = nwm.s3_nwm3_zarr_bucket('CONUS/zarr/forcing/precip.zarr')['RAINRATE']*3600 # mm/s --> mm/hr
t2d = nwm.s3_nwm3_zarr_bucket('CONUS/zarr/forcing/t2d.zarr')['T2D']-273.15 # K --> C
accet = nwm.s3_nwm3_zarr_bucket('CONUS/zarr/ldasout.zarr')['ACCET']#*3600 # mm/s --> mm/hr
lh = nwm.s3_nwm3_zarr_bucket('CONUS/zarr/ldasout.zarr')['LH'] # W/m^2
hfx = nwm.s3_nwm3_zarr_bucket('CONUS/zarr/ldasout.zarr')['HFX'] # W/m^2

# Read FLUXNET Site Info
gdf_fluxnet_site_info_utm12n_nad83 = gpd.read_file('../out/00_proc_FLUXNET_data/info/fluxnet_site_info_az_utm12n_nad83.parquet.gzip')
gdf_fluxnet_site_info_utm12n_nad83.index = gdf_fluxnet_site_info_utm12n_nad83['__index_level_0__']
gdf_fluxnet_site_info_utm12n_nad83.index.name = 'site_id'
gdf_fluxnet_site_info_utm12n_nad83 = gdf_fluxnet_site_info_utm12n_nad83.rename(columns={'name':'site_name',
                                                                                '__index_level_0__':'site_id'})  
gdf_fluxnet_site_info_nwm_lcc = gdf_fluxnet_site_info_utm12n_nad83.to_crs(param_nwm3.crs_nwm_proj4_lcc)

savedir = os.path.join(param_nwm3.out_dir,os.path.basename(__file__).split('.')[0])
misc.makedir(savedir)

# ACCET
dfs = []
for i in gdf_fluxnet_site_info_nwm_lcc.index:
    site_code = gdf_fluxnet_site_info_nwm_lcc.loc[i, 'site_id']
    site_name = gdf_fluxnet_site_info_nwm_lcc.loc[i, 'site_name']
    print('ACCET:',site_code)
    site_x = gdf_fluxnet_site_info_nwm_lcc.loc[i, 'geometry'].x
    site_y = gdf_fluxnet_site_info_nwm_lcc.loc[i, 'geometry'].y

    df = accet.sel(x=site_x, y=site_y, method='nearest').to_dataframe()['ACCET']
    df = df.rename(site_code)
    df.index.name = 'datetime'

    dfs.append(df)
dfs = pd.concat(dfs, axis=1)
dfs.to_parquet(os.path.join(savedir,'ACCET_3H.parquet.gzip'),compression='gzip')

# ET
print('Converting ACCET to ET')
dfs = nwm.accet_to_et_df(dfs)
dfs.to_parquet(os.path.join(savedir,'ET_3H.parquet.gzip'),compression='gzip')
dfs.resample('D').sum().to_parquet(os.path.join(savedir,'ET_D.parquet.gzip'),compression='gzip')
dfs.resample('ME').sum().to_parquet(os.path.join(savedir,'ET_M.parquet.gzip'),compression='gzip')

# # LH
# dfs = []
# for i in gdf_fluxnet_site_info_nwm_lcc.index:
#     site_code = gdf_fluxnet_site_info_nwm_lcc.loc[i, 'site_id']
#     site_name = gdf_fluxnet_site_info_nwm_lcc.loc[i, 'site_name']
#     print('LH:',site_code)
#     site_x = gdf_fluxnet_site_info_nwm_lcc.loc[i, 'geometry'].x
#     site_y = gdf_fluxnet_site_info_nwm_lcc.loc[i, 'geometry'].y

#     df = lh.sel(x=site_x, y=site_y, method='nearest').to_dataframe()['LH']
#     df = df.rename(site_code)
#     df.index.name = 'datetime'

#     dfs.append(df)
# dfs = pd.concat(dfs, axis=1)
# dfs.to_parquet(os.path.join(savedir,'LH_3H.parquet.gzip'),compression='gzip')
# dfs.resample('D').sum().to_parquet(os.path.join(savedir,'LH_D.parquet.gzip'),compression='gzip')

# # HFX
# dfs = []
# for i in gdf_fluxnet_site_info_nwm_lcc.index:
#     site_code = gdf_fluxnet_site_info_nwm_lcc.loc[i, 'site_id']
#     site_name = gdf_fluxnet_site_info_nwm_lcc.loc[i, 'site_name']
#     print('HFX:',site_code)
#     site_x = gdf_fluxnet_site_info_nwm_lcc.loc[i, 'geometry'].x
#     site_y = gdf_fluxnet_site_info_nwm_lcc.loc[i, 'geometry'].y

#     df = hfx.sel(x=site_x, y=site_y, method='nearest').to_dataframe()['HFX']
#     df = df.rename(site_code)
#     df.index.name = 'datetime'

#     dfs.append(df)
# dfs = pd.concat(dfs, axis=1)
# dfs.to_parquet(os.path.join(savedir,'HFX_3H.parquet.gzip'),compression='gzip')
# dfs.resample('D').sum().to_parquet(os.path.join(savedir,'HFX_D.parquet.gzip'),compression='gzip')

# PRECIP
dfs = []
for i in gdf_fluxnet_site_info_nwm_lcc.index:
    site_code = gdf_fluxnet_site_info_nwm_lcc.loc[i, 'site_id']
    site_name = gdf_fluxnet_site_info_nwm_lcc.loc[i, 'site_name']
    print('PRECIP:',site_code)
    site_x = gdf_fluxnet_site_info_nwm_lcc.loc[i, 'geometry'].x
    site_y = gdf_fluxnet_site_info_nwm_lcc.loc[i, 'geometry'].y

    df = precip.sel(x=site_x, y=site_y, method='nearest').to_dataframe()['RAINRATE']
    df = df.rename(site_code)
    df.index.name = 'datetime'

    dfs.append(df)

dfs = pd.concat(dfs, axis=1)
dfs.to_parquet(os.path.join(savedir,'PRECIP_H.parquet.gzip'),compression='gzip')
dfs.resample('D').sum().to_parquet(os.path.join(savedir,'PRECIP_D.parquet.gzip'),compression='gzip')
dfs.resample('ME').sum().to_parquet(os.path.join(savedir,'PRECIP_M.parquet.gzip'),compression='gzip')


# T2D
dfs = []
for i in gdf_fluxnet_site_info_nwm_lcc.index:
    site_code = gdf_fluxnet_site_info_nwm_lcc.loc[i, 'site_id']
    site_name = gdf_fluxnet_site_info_nwm_lcc.loc[i, 'site_name']
    print('T2D:',site_code)
    site_x = gdf_fluxnet_site_info_nwm_lcc.loc[i, 'geometry'].x
    site_y = gdf_fluxnet_site_info_nwm_lcc.loc[i, 'geometry'].y

    df = t2d.sel(x=site_x, y=site_y, method='nearest').to_dataframe()['T2D']
    df = df.rename(site_code)
    df.index.name = 'datetime'

    dfs.append(df)
dfs = pd.concat(dfs, axis=1)
dfs.to_parquet(os.path.join(savedir,'T2D_H.parquet.gzip'),compression='gzip')
dfs.resample('D').mean().to_parquet(os.path.join(savedir,'T2D_D.parquet.gzip'),compression='gzip')
dfs.resample('ME').mean().to_parquet(os.path.join(savedir,'T2D_M.parquet.gzip'),compression='gzip')

