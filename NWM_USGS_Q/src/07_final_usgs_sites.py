import os
import geopandas as gpd
import pandas as pd
import param_nwm3
import nwm
import misc

savedir = os.path.join(param_nwm3.out_dir,os.path.basename(__file__).split('.')[0])
misc.makedir(savedir)

hourly = gpd.read_parquet('../out/06A_add_attributes/parquet/hourly_2003_2022_utm12n_nad83.parquet.gzip')
daily = gpd.read_parquet('../out/06A_add_attributes/parquet/daily_2003_2022_utm12n_nad83.parquet.gzip')

gdf_site_info = gpd.read_parquet('../out/05_data_availability/usgs_q_info_flagged_utm12n_nad83.parquet.gzip')

gdf_site_info = gdf_site_info[(gdf_site_info['site_no'].isin(daily['USGS_ID'])) | (gdf_site_info['site_no'].isin(hourly['USGS_ID']))]
gdf_site_info.loc[(gdf_site_info['site_no'].isin(daily['USGS_ID'])),'daily_sel'] = 1 # daily and monthly are the same
gdf_site_info.loc[(gdf_site_info['site_no'].isin(hourly['USGS_ID'])),'hourly_sel'] = 1

gdf_site_info['sel_years_HA'] = hourly['sel_years_HA']
gdf_site_info['sel_years_DA'] = daily['sel_years_DA']
gdf_site_info['M_BASIN_HA'] = hourly['M_BASIN_ABR']
gdf_site_info['M_BASIN_DA'] = daily['M_BASIN_ABR']
gdf_site_info['M_BASIN_DA'] = gdf_site_info['M_BASIN_DA'].where(gdf_site_info['M_BASIN_DA'].notnull(),gdf_site_info['M_BASIN_HA'])
gdf_site_info = gdf_site_info.rename(columns={'M_BASIN_DA':'M_BASIN'})
gdf_site_info = gdf_site_info.drop(columns=['M_BASIN_HA'])
gdf_site_info['nid_ID'] = gdf_site_info['nid_ID'].astype(str)

gdf_site_basins = []

for i in gdf_site_info.index:
    print(i)
    station_id = gdf_site_info.loc[i,'site_no']
    basin = nwm.get_basin_by_usgsgageid(station_id).to_crs(param_nwm3.crs_nwm_proj4_lcc)
    gdf_site_info.loc[i,'nldi_area_km2'] = basin.area.values[0]/1E06
    gdf_site_basins.append(basin)

gdf_site_basins = gpd.GeoDataFrame(pd.concat(gdf_site_basins))
gdf_site_basins = gdf_site_basins.to_crs(param_nwm3.crs_utm12n_nad83)
gdf_site_basins['site_no'] = gdf_site_basins.index.str.split('-').str[1]

# Save USGS Gage Locations
gdf_site_info.to_parquet(os.path.join(savedir,'usgs_q_info_flagged_utm12n_nad83.parquet.gzip'),compression='gzip')
gdf_site_info.to_file(os.path.join(savedir,'usgs_q_info_flagged_utm12n_nad83.gpkg'))
gdf_site_info.to_file(os.path.join(savedir,'usgs_q_info_flagged_utm12n_nad83.shp'))
gdf_site_info.to_csv(os.path.join(savedir,'usgs_q_info_flagged_utm12n_nad83.csv'))

# Save NLDI Basins
gdf_site_basins.to_parquet(os.path.join(savedir,'usgs_q_info_flagged_utm12n_nad83_basins.parquet.gzip'),compression='gzip')
gdf_site_basins.to_file(os.path.join(savedir,'usgs_q_info_flagged_utm12n_nad83_basins.gpkg'))
gdf_site_basins.to_file(os.path.join(savedir,'usgs_q_info_flagged_utm12n_nad83_basins.shp'))
gdf_site_basins.to_csv(os.path.join(savedir,'usgs_q_info_flagged_utm12n_nad83_basins.csv'))


