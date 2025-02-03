import os

import xarray as xr
import numpy as np
import pandas as pd
import geopandas as gpd
import datetime

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as ticker
from matplotlib.ticker import FuncFormatter

import param_nwm3
import misc
import eval_metrics_fluxnet as emf
import nwm_static_param_dict as nspd

import seaborn as sns

def calc_metrics(df_obs,df_sim,
                 df_obs_p,df_sim_p,
                 df_obs_t2d,df_sim_t2d,
                 season_months,start_wy,end_wy):

    df_obs = df_obs.copy()
    df_sim = df_sim.copy()
    df_obs_p = df_obs_p.copy()
    df_sim_p = df_sim_p.copy()
    df_obs_t2d = df_obs_t2d.copy()
    df_sim_t2d = df_sim_t2d.copy()

    # Select Season
    df_obs = df_obs[df_obs.index.month.isin(season_months)]
    df_sim = df_sim[df_sim.index.month.isin(season_months)]
    df_obs_p = df_obs_p[df_obs_p.index.month.isin(season_months)]
    df_sim_p = df_sim_p[df_sim_p.index.month.isin(season_months)]
    df_obs_t2d = df_obs_t2d[df_obs_t2d.index.month.isin(season_months)]
    df_sim_t2d = df_sim_t2d[df_sim_t2d.index.month.isin(season_months)]

    # Calculate Seasonal Statistics (by Water Year)
    df_obs = emf.process_df(df_obs,start_wy,end_wy)
    df_sim = emf.process_df(df_sim,start_wy,end_wy)
    df_obs_p = emf.process_df(df_obs_p,start_wy,end_wy)
    df_sim_p = emf.process_df(df_sim_p,start_wy,end_wy)
    df_obs_t2d = emf.process_df(df_obs_t2d,start_wy,end_wy)
    df_sim_t2d = emf.process_df(df_sim_t2d,start_wy,end_wy)

    

    df_obs_seasonal = emf.groupby_wy_sum(df_obs)
    df_sim_seasonal = emf.groupby_wy_sum(df_sim)
    df_obs_p_seasonal = emf.groupby_wy_sum(df_obs_p)
    df_sim_p_seasonal = emf.groupby_wy_sum(df_sim_p)
    df_obs_t2d_seasonal = emf.groupby_wy_mean(df_obs_t2d)
    df_sim_t2d_seasonal = emf.groupby_wy_mean(df_sim_t2d)


    # P
    mean_obs_p = df_obs_p_seasonal.mean().values[0]
    mean_sim_p = df_sim_p_seasonal.mean().values[0]
    delta_p = mean_sim_p - mean_obs_p

    # ET
    mean_obs = df_obs_seasonal.mean().values[0]
    mean_sim = df_sim_seasonal.mean().values[0]
    delta_et = mean_sim - mean_obs

    # T2D
    mean_obs_t2d = df_obs_t2d_seasonal.mean().values[0]
    mean_sim_t2d = df_sim_t2d_seasonal.mean().values[0]
    delta_t2d = mean_sim_t2d - mean_obs_t2d


    # Calculate ET/P Ratio
    etp_sim = (df_sim_seasonal.sum() / df_sim_p_seasonal.sum()).values[0]
    etp_obs = (df_obs_seasonal.sum() / df_obs_p_seasonal.sum()).values[0]

    # Select Last column (Last column has station data)
    df_obs = df_obs.iloc[:,-1]
    df_sim = df_sim.iloc[:,-1]

    nse=emf.NSE(df_obs,df_sim)
    nnse=emf.NNSE(df_obs,df_sim)
    rmse=emf.RMSE(df_obs,df_sim)
    pbias=emf.PBIAS(df_obs,df_sim)
    bias=emf.BIAS(df_obs,df_sim)
    pearsonr=emf.PEARSON(df_obs,df_sim)




    metrics_dict = {'NSE':round(nse,2),
                    'NNSE':round(nnse,2),
                    'RMSE':round(rmse,2),
                    'PBIAS':round(pbias,2),
                    'BIAS':round(bias,2),
                    'PEARSON':round(pearsonr,2),
                    'Mean_P_Obs':round(mean_obs_p,2),
                    'Mean_P_Sim':round(mean_sim_p,2),
                    'Delta_P':round(delta_p,2),
                    'Mean_ET_Obs':round(mean_obs,2),
                    'Mean_ET_Sim':round(mean_sim,2),
                    'Delta_ET':round(delta_et,2),
                    'Mean_T2D_Obs':round(mean_obs_t2d,2),
                    'Mean_T2D_Sim':round(mean_sim_t2d,2),
                    'Delta_T2D':round(delta_t2d,2),
                    'ETP_Sim':round(etp_sim,2),
                    'ETP_Obs':round(etp_obs,2)}
    
    return metrics_dict


# Output Directory
savedir = os.path.join(param_nwm3.out_dir,os.path.basename(__file__).split('.')[0])
savedir_fig = os.path.join(param_nwm3.out_dir,'figures',os.path.basename(__file__).split('.')[0])
misc.makedir(savedir)
misc.makedir(savedir_fig)

# FLUXNET Site Info
# Read FLUXNET Site Info
gdf_fluxnet_site_info_utm12n_nad83 = gpd.read_file('../out/00_proc_FLUXNET_data/info/fluxnet_site_info_az_utm12n_nad83.parquet.gzip')
# gdf_fluxnet_site_info_utm12n_nad83.index = gdf_fluxnet_site_info_utm12n_nad83['__index_level_0__']
# gdf_fluxnet_site_info_utm12n_nad83.index.name = 'site_id'
# gdf_fluxnet_site_info_utm12n_nad83 = gdf_fluxnet_site_info_utm12n_nad83.rename(columns={'name':'site_name',
#                                                                                 '__index_level_0__':'site_id'})

# NWM Static Properties
ds_nwm_hgt_az_buffered_utm12n_nad83 = xr.open_dataset(param_nwm3.nc_nwm_hgt_az_buffered_utm12n_nad83)['HGT']
ds_nwm_isltyp_az_buffered_utm12n_nad83 = xr.open_dataset(param_nwm3.nc_nwm_isltyp_az_buffered_utm12n_nad83)['ISLTYP']
ds_nwm_ivgtyp_az_buffered_utm12n_nad83 = xr.open_dataset(param_nwm3.nc_nwm_ivgtyp_az_buffered_utm12n_nad83)['IVGTYP']

# Merged Basins
gdf_m_basins_az_utm12n_nad83 = gpd.read_file(param_nwm3.shp_az_simplified_watersheds_utm12n_nad83)

# Data Directory
fluxnet_data_dir = '../out/00_proc_FLUXNET_data/parquet'
nwm_data_dir = '../out/01_download_NWM_ET'

monthly_data_dict = {
    'ET':{'FLUXNET_GAP':os.path.join(fluxnet_data_dir,'fluxnet_ET_F_MDS_monthly.parquet.gzip'),
          'FLUXNET_CORR':os.path.join(fluxnet_data_dir,'fluxnet_ET_CORR_monthly.parquet.gzip'),
          'NWM':os.path.join(nwm_data_dir,'ET_M.parquet.gzip')},
    'P':{'FLUXNET':os.path.join(fluxnet_data_dir,'fluxnet_P_F_daily.parquet.gzip'),
            'NWM':os.path.join(nwm_data_dir,'PRECIP_M.parquet.gzip')},
    'T2D':{'FLUXNET':os.path.join(fluxnet_data_dir,'fluxnet_TA_F_MDS_monthly.parquet.gzip'),
           'NWM':os.path.join(nwm_data_dir,'T2D_M.parquet.gzip')}
}


daily_data_dict = {
    'ET':{'FLUXNET_GAP':os.path.join(fluxnet_data_dir,'fluxnet_ET_F_MDS_daily.parquet.gzip'),
          'FLUXNET_CORR':os.path.join(fluxnet_data_dir,'fluxnet_ET_CORR_daily.parquet.gzip'),
          'NWM':os.path.join(nwm_data_dir,'ET_D.parquet.gzip')},
    'P':{'FLUXNET':os.path.join(fluxnet_data_dir,'fluxnet_P_F_daily.parquet.gzip'),
            'NWM':os.path.join(nwm_data_dir,'PRECIP_D.parquet.gzip')},
    'T2D':{'FLUXNET':os.path.join(fluxnet_data_dir,'fluxnet_TA_F_MDS_daily.parquet.gzip'),
           'NWM':os.path.join(nwm_data_dir,'T2D_D.parquet.gzip')}
}

hourly_data_dict = {
    'ET':{'FLUXNET_GAP':os.path.join(fluxnet_data_dir,'fluxnet_ET_F_MDS_hourly.parquet.gzip'),
          'FLUXNET_CORR':os.path.join(fluxnet_data_dir,'fluxnet_ET_CORR_hourly.parquet.gzip'),
          'NWM':os.path.join(nwm_data_dir,'ET_3H.parquet.gzip')},
    'P':{'FLUXNET':os.path.join(fluxnet_data_dir,'fluxnet_P_F_daily.parquet.gzip'),
            'NWM':os.path.join(nwm_data_dir,'PRECIP_H.parquet.gzip')},
    'T2D':{'FLUXNET':os.path.join(fluxnet_data_dir,'fluxnet_TA_F_MDS_hourly.parquet.gzip'),
           'NWM':os.path.join(nwm_data_dir,'T2D_H.parquet.gzip')}
}

frequencies = [
    # 'hourly',
    # 'daily',
    'monthly'
]


sel_vars = [
    'ET'
]


seasons = {
           'WY':{'months':[1,2,3,4,5,6,7,8,9,10,11,12]},   # Jan - Dec
           'W':{'months':[11,12,1,2,3]}, # Nov - Mar
           'S':{'months':[7,8,9]} # Jul - Sep
          }


# Write Static Properties
gdf_fluxnet_site_info_utm12n_nad83 = gpd.sjoin(gdf_fluxnet_site_info_utm12n_nad83,gdf_m_basins_az_utm12n_nad83,how='left',predicate='within')
gdf_fluxnet_site_info_utm12n_nad83 = gdf_fluxnet_site_info_utm12n_nad83.rename(columns={'NAME':'M_BASIN',
                                                                                      'NAME_ABR':'M_BASIN_ABR'})
gdf_fluxnet_site_info_utm12n_nad83 = gdf_fluxnet_site_info_utm12n_nad83.rename(columns={'__index_level_0__':'code'})
gdf_fluxnet_site_info_utm12n_nad83.index = gdf_fluxnet_site_info_utm12n_nad83['code']
gdf_fluxnet_site_info_utm12n_nad83 = gdf_fluxnet_site_info_utm12n_nad83.drop(columns=['index_right','area_m2'])

# Write Static NWM Properties
gdfs_static_data = [ds_nwm_hgt_az_buffered_utm12n_nad83,
                    ds_nwm_isltyp_az_buffered_utm12n_nad83,
                    ds_nwm_ivgtyp_az_buffered_utm12n_nad83]
dfs_static_data = []
for gdf_static_data in gdfs_static_data:
    var_name = gdf_static_data.name
    df_static_data = emf.get_static_nwm_data(gdf_static_data,gdf_fluxnet_site_info_utm12n_nad83,var_name)
    dfs_static_data.append(df_static_data)
dfs_static_data = pd.concat(dfs_static_data,axis=1)
dfs_static_data = dfs_static_data.rename(columns={'HGT':'NWM_Elevation_m',
                                                    'ISLTYP':'NWM_ISLTYP',
                                                    'IVGTYP':'NWM_IVGTYP'})
dfs_static_data['NWM_ISLTYP_NAME'] = dfs_static_data['NWM_ISLTYP'].map({k:v[0] for k,v in nspd.nwm_soil_name_dict.items()})
dfs_static_data['NWM_ISLTYP_ABR'] = dfs_static_data['NWM_ISLTYP'].map({k:v[1] for k,v in nspd.nwm_soil_name_dict.items()})
dfs_static_data['NWM_IVGTYP_NAME'] = dfs_static_data['NWM_IVGTYP'].map({k:v[0] for k,v in nspd.nwm_land_name_dict.items()})
dfs_static_data['NWM_IVGTYP_ABR'] = dfs_static_data['NWM_IVGTYP'].map({k:v[1] for k,v in nspd.nwm_land_name_dict.items()})
dfs_static_data['NWM_KG_ABR'] = dfs_static_data['NWM_IVGTYP'].map({k:v[1] for k,v in nspd.koppen_name_dict.items()})

gdf_fluxnet_site_info_utm12n_nad83 = pd.concat([gdf_fluxnet_site_info_utm12n_nad83,dfs_static_data],axis=1)
gdf_fluxnet_site_info_utm12n_nad83 = gdf_fluxnet_site_info_utm12n_nad83.rename(columns={'elev':'FLUXNET_Elevation_m'})
gdf_fluxnet_site_info_utm12n_nad83 = gdf_fluxnet_site_info_utm12n_nad83.astype({'FLUXNET_Elevation_m':float,
                                                                                'lat':float,
                                                                                'lon':float})
gdf_fluxnet_site_info_utm12n_nad83['Delta_Elevation_m'] = gdf_fluxnet_site_info_utm12n_nad83['NWM_Elevation_m'] - gdf_fluxnet_site_info_utm12n_nad83['FLUXNET_Elevation_m']


# Calculate Metrics
for frequency in frequencies:
    if frequency == 'hourly':
        data_dict = hourly_data_dict
    elif frequency == 'daily':
        data_dict = daily_data_dict
        # data_dict = hourly_data_dict
    elif frequency == 'monthly':
        # data_dict = daily_data_dict
        data_dict = monthly_data_dict

    gdf_fluxnet_site_info_utm12n_nad83 = gdf_fluxnet_site_info_utm12n_nad83.copy()

    for variable in sel_vars:
        print(frequency,variable)
        # Read Data
        df_fluxnet = pd.read_parquet(data_dict[variable]['FLUXNET_GAP'])
        df_fluxnet_corr = pd.read_parquet(data_dict[variable]['FLUXNET_CORR'])
        df_nwm = pd.read_parquet(data_dict[variable]['NWM'])


        for station in gdf_fluxnet_site_info_utm12n_nad83.index:
            df_fluxnet_station = df_fluxnet[[station]]
            df_fluxnet_corr_station = df_fluxnet_corr[[station]]
            df_nwm_station = df_nwm[[station]]
            igbp = gdf_fluxnet_site_info_utm12n_nad83.loc[station,'igbp']

            start_wy,end_wy,n_wy = emf.data_availabilty(df_fluxnet_station)
            df_nwm_station.index = df_nwm_station.index.shift(-3,freq='h')

            if frequency == 'hourly':
                df_fluxnet_station = df_fluxnet_station.dropna()
                # df_fluxnet_station = df_fluxnet_station.resample('3h',label='right').sum()
                df_fluxnet_station = df_fluxnet_station.resample('3h').sum()

                df_fluxnet_corr_station = df_fluxnet_corr_station.dropna()
                df_fluxnet_corr_station = df_fluxnet_corr_station.resample('3h').sum()


            elif frequency == 'daily':
                df_fluxnet_station = df_fluxnet_station.dropna()
                # df_fluxnet_station.index = df_fluxnet_station.index.normalize()
                df_fluxnet_station = df_fluxnet_station.resample('D').sum()

                df_fluxnet_corr_station = df_fluxnet_corr_station.dropna()
                # df_fluxnet_corr_station.index = df_fluxnet_corr_station.index.normalize()
                df_fluxnet_corr_station = df_fluxnet_corr_station.resample('D').sum()

                df_nwm_station = df_nwm_station.dropna()
                # df_nwm_station.index = df_nwm_station.index.normalize()
                df_nwm_station = df_nwm_station.resample('D').sum()

            elif frequency == 'monthly':
                df_fluxnet_station = df_fluxnet_station.dropna()
                # df_fluxnet_station.index = df_fluxnet_station.index.normalize()
                # df_fluxnet_station.index = df_fluxnet_station.index.to_period('M')
                df_fluxnet_station = df_fluxnet_station.resample('ME').sum()

                df_fluxnet_corr_station = df_fluxnet_corr_station.dropna()
                # df_fluxnet_corr_station.index = df_fluxnet_corr_station.index.normalize()
                # df_fluxnet_corr_station.index = df_fluxnet_corr_station.index.to_period('M')
                df_fluxnet_corr_station = df_fluxnet_corr_station.resample('ME').sum()

                df_nwm_station = df_nwm_station.dropna()
                # df_nwm_station.index = df_nwm_station.index.normalize()
                # df_nwm_station.index = df_nwm_station.index.to_period('M')
                df_nwm_station = df_nwm_station.resample('ME').sum()

            df_nwm_station = df_nwm_station.loc[df_fluxnet_station.index]


            # Convert to US/Arizona Timezone
            df_fluxnet_station.index = df_fluxnet_station.index.tz_localize('UTC').tz_convert('US/Arizona')
            df_fluxnet_corr_station.index = df_fluxnet_corr_station.index.tz_localize('UTC').tz_convert('US/Arizona')
            df_nwm_station.index = df_nwm_station.index.tz_localize('UTC').tz_convert('US/Arizona')


            # Read P
            df_fluxnet_precip_station = pd.read_parquet(data_dict['P']['FLUXNET'])[[station]]
            df_fluxnet_precip_station.index = df_fluxnet_precip_station.index.tz_localize('UTC').tz_convert('US/Arizona')

            df_nwm_precip_station = pd.read_parquet(data_dict['P']['NWM'])[[station]]
            df_nwm_precip_station.index = df_nwm_precip_station.index.tz_localize('UTC').tz_convert('US/Arizona')

            # Read T 
            df_fluxnet_t2d_station = pd.read_parquet(data_dict['T2D']['FLUXNET'])[[station]]
            df_fluxnet_t2d_station.index = df_fluxnet_t2d_station.index.tz_localize('UTC').tz_convert('US/Arizona')

            df_nwm_t2d_station = pd.read_parquet(data_dict['T2D']['NWM'])[[station]]
            df_nwm_t2d_station.index = df_nwm_t2d_station.index.tz_localize('UTC').tz_convert('US/Arizona')
            df_nwm_t2d_station = df_nwm_t2d_station.dropna()


            for season in seasons.keys():
                season_months = seasons[season]['months']

                # GAP
                fluxnet_metrics = calc_metrics(df_obs=df_fluxnet_station,
                                                df_sim=df_nwm_station,
                                                df_obs_p=df_fluxnet_precip_station,
                                                df_sim_p=df_nwm_precip_station,
                                                df_obs_t2d=df_fluxnet_t2d_station,
                                                df_sim_t2d=df_nwm_t2d_station,
                                                season_months=season_months,
                                                start_wy=start_wy,
                                                end_wy=end_wy)

                # CORR
                fluxnet_corr_metrics = calc_metrics(df_obs=df_fluxnet_corr_station,
                                                    df_sim=df_nwm_station,
                                                    df_obs_p=df_fluxnet_precip_station,
                                                    df_sim_p=df_nwm_precip_station,
                                                    df_obs_t2d=df_fluxnet_t2d_station,
                                                    df_sim_t2d=df_nwm_t2d_station,
                                                    season_months=season_months,
                                                    start_wy=start_wy,
                                                    end_wy=end_wy)




                # Save Metrics & Statistics

                # GAP
                metrics_dict = fluxnet_metrics
                gdf_fluxnet_site_info_utm12n_nad83.loc[station,f'{variable}_NSE_{season}'] = metrics_dict['NSE']
                gdf_fluxnet_site_info_utm12n_nad83.loc[station,f'{variable}_NNSE_{season}'] = metrics_dict['NNSE']
                gdf_fluxnet_site_info_utm12n_nad83.loc[station,f'{variable}_RMSE_{season}'] = metrics_dict['RMSE']
                gdf_fluxnet_site_info_utm12n_nad83.loc[station,f'{variable}_PBIAS_{season}'] = metrics_dict['PBIAS']
                gdf_fluxnet_site_info_utm12n_nad83.loc[station,f'{variable}_BIAS_{season}'] = metrics_dict['BIAS']
                gdf_fluxnet_site_info_utm12n_nad83.loc[station,f'{variable}_PEARSON_{season}'] = metrics_dict['PEARSON']
                gdf_fluxnet_site_info_utm12n_nad83.loc[station,f'FLUXNET_ETP_{season}'] = metrics_dict['ETP_Obs']
                gdf_fluxnet_site_info_utm12n_nad83.loc[station,f'NWM_ETP_{season}'] = metrics_dict['ETP_Sim']
                gdf_fluxnet_site_info_utm12n_nad83.loc[station,f'FLUXNET_ET_{season}'] = metrics_dict['Mean_ET_Obs']
                gdf_fluxnet_site_info_utm12n_nad83.loc[station,f'NWM_ET_{season}'] = metrics_dict['Mean_ET_Sim']
                gdf_fluxnet_site_info_utm12n_nad83.loc[station,f'Delta_ET_{season}'] = metrics_dict['Delta_ET']
                gdf_fluxnet_site_info_utm12n_nad83.loc[station,f'FLUXNET_P_{season}'] = metrics_dict['Mean_P_Obs']
                gdf_fluxnet_site_info_utm12n_nad83.loc[station,f'NWM_P_{season}'] = metrics_dict['Mean_P_Sim']
                gdf_fluxnet_site_info_utm12n_nad83.loc[station,f'Delta_P_{season}'] = metrics_dict['Delta_P']
                gdf_fluxnet_site_info_utm12n_nad83.loc[station,f'FLUXNET_T2D_{season}'] = metrics_dict['Mean_T2D_Obs']
                gdf_fluxnet_site_info_utm12n_nad83.loc[station,f'NWM_T2D_{season}'] = metrics_dict['Mean_T2D_Sim']
                gdf_fluxnet_site_info_utm12n_nad83.loc[station,f'Delta_T2D_{season}'] = metrics_dict['Delta_T2D']

                # CORR
                metrics_dict = fluxnet_corr_metrics
                gdf_fluxnet_site_info_utm12n_nad83.loc[station,f'{variable}_NSE_C_{season}'] = metrics_dict['NSE']
                gdf_fluxnet_site_info_utm12n_nad83.loc[station,f'{variable}_NNSE_C_{season}'] = metrics_dict['NNSE']
                gdf_fluxnet_site_info_utm12n_nad83.loc[station,f'{variable}_RMSE_C_{season}'] = metrics_dict['RMSE']
                gdf_fluxnet_site_info_utm12n_nad83.loc[station,f'{variable}_PBIAS_C_{season}'] = metrics_dict['PBIAS']
                gdf_fluxnet_site_info_utm12n_nad83.loc[station,f'{variable}_BIAS_C_{season}'] = metrics_dict['BIAS']
                gdf_fluxnet_site_info_utm12n_nad83.loc[station,f'{variable}_PEARSON_C_{season}'] = metrics_dict['PEARSON']
                gdf_fluxnet_site_info_utm12n_nad83.loc[station,f'FLUXNET_ETP_C_{season}'] = metrics_dict['ETP_Obs']
                gdf_fluxnet_site_info_utm12n_nad83.loc[station,f'NWM_ETP_C_{season}'] = metrics_dict['ETP_Sim']
                gdf_fluxnet_site_info_utm12n_nad83.loc[station,f'FLUXNET_ET_C_{season}'] = metrics_dict['Mean_ET_Obs']
                gdf_fluxnet_site_info_utm12n_nad83.loc[station,f'NWM_ET_C_{season}'] = metrics_dict['Mean_ET_Sim']
                gdf_fluxnet_site_info_utm12n_nad83.loc[station,f'Delta_ET_C_{season}'] = metrics_dict['Delta_ET']
                gdf_fluxnet_site_info_utm12n_nad83.loc[station,f'FLUXNET_P_C_{season}'] = metrics_dict['Mean_P_Obs']
                gdf_fluxnet_site_info_utm12n_nad83.loc[station,f'NWM_P_C_{season}'] = metrics_dict['Mean_P_Sim']
                gdf_fluxnet_site_info_utm12n_nad83.loc[station,f'Delta_P_C_{season}'] = metrics_dict['Delta_P']
                gdf_fluxnet_site_info_utm12n_nad83.loc[station,f'FLUXNET_T2D_C_{season}'] = metrics_dict['Mean_T2D_Obs']
                gdf_fluxnet_site_info_utm12n_nad83.loc[station,f'NWM_T2D_C_{season}'] = metrics_dict['Mean_T2D_Sim']
                gdf_fluxnet_site_info_utm12n_nad83.loc[station,f'Delta_T2D_C_{season}'] = metrics_dict['Delta_T2D']



            # TODO: Add plots here

            df_fluxnet_station = df_fluxnet_station.loc[:,[station]]
            df_fluxnet_corr_station = df_fluxnet_corr_station.loc[:,[station]]
            df_nwm_station = df_nwm_station.loc[:,[station]]
            
            # Plot 1: Compare GAP and CORR Data
            # Subplot 1: Full Time Series
            print(station)
            fig,axs = plt.subplots(1, 2, figsize=(10, 3), gridspec_kw={'width_ratios': [2, 1]})
            ax=axs[0]
            plot_df = pd.concat([df_nwm_station,df_fluxnet_station,df_fluxnet_corr_station],axis=1)
            plot_df.columns = ['NWM','FLUXNET [GAP]','FLUXNET [CORR]']
            plot_df.plot(ax=ax,style=['k-','r--','b--'],lw=0.8)
            ax.grid(True,color='gray',linestyle='--',linewidth=0.5)
            ax.set_title('Site: {} | Land cover: {} \n(WY {}-{} [{}])'.format(station,igbp,start_wy,end_wy,n_wy))
            if frequency == 'hourly':
                ax.set_ylabel('ET (mm/3-hr)')
            elif frequency == 'daily':
                ax.set_ylabel('ET (mm/day)')
            elif frequency == 'monthly':
                ax.set_ylabel('ET (mm/month)')
            ax.set_xlabel(None)

            # Subplot 2: Scatter Plot
            ax=axs[1]
            sns.scatterplot(data=plot_df,x='NWM',y='FLUXNET [GAP]',s=1,ax=ax,color='r',label='FLUXNET [GAP]')
            sns.scatterplot(data=plot_df,x='NWM',y='FLUXNET [CORR]',s=1,ax=ax,color='b',label='FLUXNET [CORR]')
            ax.grid(True,color='gray',linestyle='--',linewidth=0.8)
            ax.set_axisbelow(True)
            ax.set_aspect('equal')
            ax.xaxis.set_major_locator(ticker.MaxNLocator(6))
            ax.yaxis.set_major_locator(ticker.MaxNLocator(6))
            ax.tick_params(axis='both', which='major', labelsize=12)
            ax.tick_params(axis='x', which='major', labelsize=12,rotation=90)
            # Get the current limits
            xmin, xmax = ax.get_xlim()
            ymin, ymax = ax.get_ylim()

            # Set the same limits for both axes
            limit_max = max(xmax, ymax)
            limit_min = min(xmin, ymin)
            ax.set_xlim(limit_min, limit_max)
            ax.set_ylim(limit_min, limit_max)
            ax.plot([limit_min, limit_max], [limit_min, limit_max], ls='--', color='k', lw=0.6)

            # NWM vs FLUXNET [GAP]
            z=np.polyfit(plot_df['NWM'],plot_df['FLUXNET [GAP]'],1)
            p=np.poly1d(z)
            ax.plot(plot_df['NWM'],p(plot_df['NWM']),color='r',ls='--',lw=0.8)


            # NWM vs FLUXNET [CORR]
            z=np.polyfit(plot_df['NWM'],plot_df['FLUXNET [CORR]'],1)
            p=np.poly1d(z)
            ax.plot(plot_df['NWM'],p(plot_df['NWM']),color='b',ls='--',lw=0.8)

            if frequency=='hourly':
                ax.set_xlabel('NWM (mm/3-hr)')
                ax.set_ylabel('FLUXNET (mm/3-hr)')
            elif frequency=='daily':
                ax.set_xlabel('NWM (mm/day)')
                ax.set_ylabel('FLUXNET (mm/day)')
            elif frequency=='monthly':
                ax.set_xlabel('NWM (mm/month)')
                ax.set_ylabel('FLUXNET (mm/month)')

            ax.set_title('Site: {} | Land cover: {} \n(WY {}-{} [{}])'.format(station,igbp,start_wy,end_wy,n_wy))
            
            box = plt.gcf().add_axes([0.9, 0.1, 0.1, 0.8], frame_on=False)
            box.axis('off')
            textstr = f"GAP\nNNSE: {fluxnet_metrics['NNSE']}\nRMSE: {fluxnet_metrics['RMSE']}\nPBIAS: {fluxnet_metrics['PBIAS']}\nBias: {fluxnet_metrics['BIAS']}\nPearson: {fluxnet_metrics['PEARSON']}"
            box.text(0.1, 0.95, textstr, fontsize=10, verticalalignment='top',color='r')

            textstr = f"CORR\nNNSE: {fluxnet_corr_metrics['NNSE']}\nRMSE: {fluxnet_corr_metrics['RMSE']}\nPBIAS: {fluxnet_corr_metrics['PBIAS']}\nBias: {fluxnet_corr_metrics['BIAS']}\nPearson: {fluxnet_corr_metrics['PEARSON']}"
            box.text(0.1, 0.50, textstr, fontsize=10, verticalalignment='top',color='b')
            
            misc.makedir(os.path.join(savedir_fig,'compare_GAP_CORR'))
            fig.savefig(os.path.join(savedir_fig,'compare_GAP_CORR',f'{frequency}_{station}.png'),dpi=300,bbox_inches='tight')
            fig.savefig(os.path.join(savedir_fig,'compare_GAP_CORR',f'{frequency}_{station}.svg'),dpi=300,bbox_inches='tight')


            # Plot 2: Compare only NWM and CORR Data
            # Subplot 1: Full Time Series
            fig,axs = plt.subplots(1, 2, figsize=(10, 3), gridspec_kw={'width_ratios': [2, 1]})
            ax=axs[0]
            plot_df = pd.concat([df_nwm_station[station],df_fluxnet_corr_station[station]],axis=1)
            plot_df.columns = ['NWM','FLUXNET']
            plot_df = plot_df[['FLUXNET','NWM']]
            plot_df.plot(ax=ax,style=['k-','--r'],lw=0.8)
            ax.grid(True,color='gray',linestyle='--',linewidth=0.5)
            ax.set_title('Site: {} | Land cover: {} \n(WY {}-{} [{}])'.format(station,igbp,start_wy,end_wy,n_wy))
            if frequency == 'hourly':
                ax.set_ylabel('ET (mm/3-hr)')
            elif frequency == 'daily':
                ax.set_ylabel('ET (mm/day)')
            elif frequency == 'monthly':
                ax.set_ylabel('ET (mm/month)')
            ax.set_xlabel(None)

            # Subplot 2: Scatter Plot
            ax=axs[1]
            sns.scatterplot(data=plot_df,x='NWM',y='FLUXNET',s=1,ax=ax,color='r',label='FLUXNET')
            ax.grid(True,color='gray',linestyle='--',linewidth=0.5)
            ax.set_axisbelow(True)
            ax.set_aspect('equal')
            ax.xaxis.set_major_locator(ticker.MaxNLocator(6))
            ax.yaxis.set_major_locator(ticker.MaxNLocator(6))
            ax.tick_params(axis='both', which='major', labelsize=12)
            ax.tick_params(axis='x', which='major', labelsize=12,rotation=90)
            # Get the current limits
            xmin, xmax = ax.get_xlim()
            ymin, ymax = ax.get_ylim()

            # Set the same limits for both axes
            limit_max = max(xmax, ymax)
            limit_min = min(xmin, ymin)
            ax.set_xlim(limit_min, limit_max)
            ax.set_ylim(limit_min, limit_max)
            ax.plot([limit_min, limit_max], [limit_min, limit_max], ls='--', color='k', lw=0.6)

            # NWM vs FLUXNET [CORR]
            z=np.polyfit(plot_df['NWM'],plot_df['FLUXNET'],1)
            p=np.poly1d(z)
            ax.plot(plot_df['NWM'],p(plot_df['NWM']),color='r',ls='--',lw=0.8)

            if frequency=='hourly':
                ax.set_xlabel('NWM (mm/3-hr)')
                ax.set_ylabel('FLUXNET (mm/3-hr)')
            elif frequency=='daily':
                ax.set_xlabel('NWM (mm/day)')
                ax.set_ylabel('FLUXNET (mm/day)')
            elif frequency=='monthly':
                ax.set_xlabel('NWM (mm/month)')
                ax.set_ylabel('FLUXNET (mm/month)')

            ax.set_title('Site: {} | Land cover: {} \n(WY {}-{} [{}])'.format(station,igbp,start_wy,end_wy,n_wy))

            box = plt.gcf().add_axes([0.9, 0.1, 0.1, 0.8], frame_on=False)
            box.axis('off')
            textstr = f"NNSE: {fluxnet_corr_metrics['NNSE']}\nRMSE: {fluxnet_corr_metrics['RMSE']}\nPBIAS: {fluxnet_corr_metrics['PBIAS']}\nBias: {fluxnet_corr_metrics['BIAS']}\nPearson: {fluxnet_corr_metrics['PEARSON']}"
            box.text(0.1, 0.50, textstr, fontsize=10, verticalalignment='top',color='r')

            misc.makedir(os.path.join(savedir_fig,'compare_CORR_NWM'))
            fig.savefig(os.path.join(savedir_fig,'compare_CORR_NWM',f'{frequency}_{station}.png'),dpi=300,bbox_inches='tight')
            fig.savefig(os.path.join(savedir_fig,'compare_CORR_NWM',f'{frequency}_{station}.svg'),dpi=300,bbox_inches='tight')

            # Plot 3: Diurnal Cycle
            if frequency == 'hourly':
                fig,ax = plt.subplots(figsize=(3,2))

                # Calc Mean
                df_fluxnet_corr_mean = df_fluxnet_corr_station.groupby(df_fluxnet_corr_station.index.hour).mean()
                df_nwm_mean = df_nwm_station.groupby(df_nwm_station.index.hour).mean()
                df_mean = pd.concat([df_fluxnet_corr_mean,df_nwm_mean],axis=1)
                df_mean.columns = ['FLUXNET','NWM']

                # Calc Std
                df_fluxnet_corr_std = df_fluxnet_corr_station.groupby(df_fluxnet_corr_station.index.hour).std()
                df_nwm_std = df_nwm_station.groupby(df_nwm_station.index.hour).std()
                df_std = pd.concat([df_fluxnet_corr_std,df_nwm_std],axis=1)
                df_std.columns = ['FLUXNET','NWM']

                # Calc 5-95 Percentile
                df_fluxnet_corr_5 = df_fluxnet_corr_station.groupby(df_fluxnet_corr_station.index.hour).quantile(0.05)
                df_nwm_5 = df_nwm_station.groupby(df_nwm_station.index.hour).quantile(0.05)
                df_5 = pd.concat([df_fluxnet_corr_5,df_nwm_5],axis=1)
                df_5.columns = ['FLUXNET','NWM']

                df_fluxnet_corr_95 = df_fluxnet_corr_station.groupby(df_fluxnet_corr_station.index.hour).quantile(0.95)
                df_nwm_95 = df_nwm_station.groupby(df_nwm_station.index.hour).quantile(0.95)
                df_95 = pd.concat([df_fluxnet_corr_95,df_nwm_95],axis=1)
                df_95.columns = ['FLUXNET','NWM']

                # Plot
                df_mean.loc[:,'FLUXNET'].plot(ax=ax,color='k',ls='-',lw=1,label='FLUXNET')
                # ax.fill_between(df_mean.index,df_mean['FLUXNET']-df_std['FLUXNET'],df_mean['FLUXNET']+df_std['FLUXNET'],color='k',alpha=0.2)
                ax.fill_between(df_mean.index,df_5['FLUXNET'],df_95['FLUXNET'],color='k',alpha=0.2)


                df_mean.loc[:,'NWM'].plot(ax=ax,color='r',ls='--',lw=1,label='NWM')
                # ax.fill_between(df_mean.index,df_mean['NWM']-df_std['NWM'],df_mean['NWM']+df_std['NWM'],color='r',alpha=0.2)
                ax.fill_between(df_mean.index,df_5['NWM'],df_95['NWM'],color='r',alpha=0.2)

                ax.set_xlim(2,23)
                ax.grid(True,color='gray',linestyle='--',linewidth=0.5)
                ax.set_axisbelow(True)
                ax.set_title('Site: {} | Land cover: {} \n(WY {}-{} [{}])'.format(station,igbp,start_wy,end_wy,n_wy))
                ax.set_xlabel('Hour of day',fontsize=12)
                ax.set_ylabel('ET (mm/3-hr)',fontsize=12)
                ax.legend(prop={'size': 8})

                box = plt.gcf().add_axes([0.9, 0.1, 0.1, 0.8], frame_on=False)
                box.axis('off')
                textstr = f"CORR\nNNSE: {fluxnet_corr_metrics['NNSE']}\nRMSE: {fluxnet_corr_metrics['RMSE']}\nPBIAS: {fluxnet_corr_metrics['PBIAS']}\nBias: {fluxnet_corr_metrics['BIAS']}\nPearson: {fluxnet_corr_metrics['PEARSON']}"
                box.text(0.2, 0.7, textstr, fontsize=10, verticalalignment='top',color='k')
                
                misc.makedir(os.path.join(savedir_fig,'diurnal_cycle'))
                fig.savefig(os.path.join(savedir_fig,'diurnal_cycle',f'{station}.png'),dpi=300,bbox_inches='tight')
                fig.savefig(os.path.join(savedir_fig,'diurnal_cycle',f'{station}.svg'),dpi=300,bbox_inches='tight')


            # Plot 3: Monthly Climatology
            if frequency == 'monthly':
                fig,ax = plt.subplots(figsize=(3,3))

                # Calc Mean
                df_fluxnet_corr_mean = df_fluxnet_corr_station.groupby(df_fluxnet_corr_station.index.month).mean()
                df_nwm_mean = df_nwm_station.groupby(df_nwm_station.index.month).mean()
                df_mean = pd.concat([df_fluxnet_corr_mean,df_nwm_mean],axis=1)
                df_mean.columns = ['FLUXNET','NWM']

                # Calc Std
                df_fluxnet_corr_std = df_fluxnet_corr_station.groupby(df_fluxnet_corr_station.index.month).std()
                df_nwm_std = df_nwm_station.groupby(df_nwm_station.index.month).std()
                df_std = pd.concat([df_fluxnet_corr_std,df_nwm_std],axis=1)
                df_std.columns = ['FLUXNET','NWM']

                # Calc 5-95 Percentile
                df_fluxnet_corr_5 = df_fluxnet_corr_station.groupby(df_fluxnet_corr_station.index.month).quantile(0.05)
                df_nwm_5 = df_nwm_station.groupby(df_nwm_station.index.month).quantile(0.05)
                df_5 = pd.concat([df_fluxnet_corr_5,df_nwm_5],axis=1)
                df_5.columns = ['FLUXNET','NWM']

                df_fluxnet_corr_95 = df_fluxnet_corr_station.groupby(df_fluxnet_corr_station.index.month).quantile(0.95)
                df_nwm_95 = df_nwm_station.groupby(df_nwm_station.index.month).quantile(0.95)
                df_95 = pd.concat([df_fluxnet_corr_95,df_nwm_95],axis=1)
                df_95.columns = ['FLUXNET','NWM']

                # Plot
                df_mean.loc[:,'FLUXNET'].plot(ax=ax,color='k',ls='-',lw=1,label='FLUXNET')
                # ax.fill_between(df_mean.index,df_mean['FLUXNET']-df_std['FLUXNET'],df_mean['FLUXNET']+df_std['FLUXNET'],color='k',alpha=0.2)
                ax.fill_between(df_mean.index,df_5['FLUXNET'],df_95['FLUXNET'],color='k',alpha=0.2)

                
                df_mean.loc[:,'NWM'].plot(ax=ax,color='r',ls='--',lw=1,label='NWM')
                # ax.fill_between(df_mean.index,df_mean['NWM']-df_std['NWM'],df_mean['NWM']+df_std['NWM'],color='r',alpha=0.2)
                ax.fill_between(df_mean.index,df_5['NWM'],df_95['NWM'],color='r',alpha=0.2)

                ax.set_xlim(1,12)
                ax.grid(True,color='gray',linestyle='--',linewidth=0.2)
                ax.set_axisbelow(True)
                # ax.set_title('Site: {} | Land cover: {} \n(WY {}-{} [{}])'.format(station,igbp,start_wy,end_wy,n_wy))
                ax.set_xlabel('Month',fontsize=12)
                ax.set_ylabel('ET (mm/month)',fontsize=12)
                # ax.legend(prop={'size': 10})
                ax.text(0.02, 0.97, f'{station}\n{start_wy}-{end_wy}', transform=ax.transAxes, fontsize=14,
                    verticalalignment='top', horizontalalignment='left', fontweight='semibold')
                ax.set_xticks(np.arange(1,13))

                box = plt.gcf().add_axes([0.9, 0.1, 0.1, 0.8], frame_on=False)
                box.axis('off')
                # textstr = f"CORR\nNNSE: {fluxnet_corr_metrics['NNSE']}\nRMSE: {fluxnet_corr_metrics['RMSE']}\nPBIAS: {fluxnet_corr_metrics['PBIAS']}\nBias: {fluxnet_corr_metrics['BIAS']}\nPearson: {fluxnet_corr_metrics['PEARSON']}"
                textstr = f"NNSE: {fluxnet_corr_metrics['NNSE']}\nRMSE: {fluxnet_corr_metrics['RMSE']}\nPBIAS: {fluxnet_corr_metrics['PBIAS']}"
                box.text(0.2, 0.7, textstr, fontsize=10, verticalalignment='top',color='k')

                misc.makedir(os.path.join(savedir_fig,'monthly_climatology'))
                fig.savefig(os.path.join(savedir_fig,'monthly_climatology',f'{station}.png'),dpi=300,bbox_inches='tight')
                fig.savefig(os.path.join(savedir_fig,'monthly_climatology',f'{station}.svg'),dpi=300,bbox_inches='tight')


            # Plot 4: Daily Climatology
            if frequency == 'daily':
                fig,ax = plt.subplots(figsize=(4,3))

                # Calc Mean
                df_fluxnet_corr_mean = df_fluxnet_corr_station.groupby(df_fluxnet_corr_station.index.dayofyear).mean()
                df_nwm_mean = df_nwm_station.groupby(df_nwm_station.index.dayofyear).mean()
                df_mean = pd.concat([df_fluxnet_corr_mean,df_nwm_mean],axis=1)
                df_mean.columns = ['FLUXNET','NWM']

                # Calc Std
                df_fluxnet_corr_std = df_fluxnet_corr_station.groupby(df_fluxnet_corr_station.index.dayofyear).std()
                df_nwm_std = df_nwm_station.groupby(df_nwm_station.index.dayofyear).std()
                df_std = pd.concat([df_fluxnet_corr_std,df_nwm_std],axis=1)
                df_std.columns = ['FLUXNET','NWM']

                # Calc 5-95 Percentile
                df_fluxnet_corr_5 = df_fluxnet_corr_station.groupby(df_fluxnet_corr_station.index.dayofyear).quantile(0.05)
                df_nwm_5 = df_nwm_station.groupby(df_nwm_station.index.dayofyear).quantile(0.05)
                df_5 = pd.concat([df_fluxnet_corr_5,df_nwm_5],axis=1)
                df_5.columns = ['FLUXNET','NWM']

                df_fluxnet_corr_95 = df_fluxnet_corr_station.groupby(df_fluxnet_corr_station.index.dayofyear).quantile(0.95)
                df_nwm_95 = df_nwm_station.groupby(df_nwm_station.index.dayofyear).quantile(0.95)
                df_95 = pd.concat([df_fluxnet_corr_95,df_nwm_95],axis=1)
                df_95.columns = ['FLUXNET','NWM']

                # Plot
                df_mean.loc[:,'FLUXNET'].plot(ax=ax,color='k',ls='-',lw=1,label='FLUXNET')
                # ax.fill_between(df_mean.index,df_mean['FLUXNET']-df_std['FLUXNET'],df_mean['FLUXNET']+df_std['FLUXNET'],color='k',alpha=0.2)
                ax.fill_between(df_mean.index,df_5['FLUXNET'],df_95['FLUXNET'],color='k',alpha=0.2)


                df_mean.loc[:,'NWM'].plot(ax=ax,color='r',ls='--',lw=1,label='NWM')
                # ax.fill_between(df_mean.index,df_mean['NWM']-df_std['NWM'],df_mean['NWM']+df_std['NWM'],color='r',alpha=0.2)
                ax.fill_between(df_mean.index,df_5['NWM'],df_95['NWM'],color='r',alpha=0.2)

                ax.set_xlim(1,366)
                ax.grid(True,color='gray',linestyle='--',linewidth=0.5)
                ax.set_axisbelow(True)
                ax.set_title('Site: {} | Land cover: {} \n(WY {}-{} [{}])'.format(station,igbp,start_wy,end_wy,n_wy))
                ax.set_xlabel('Day of year',fontsize=12)
                ax.set_ylabel('ET (mm/day)',fontsize=12)
                ax.legend(prop={'size': 10})

                box = plt.gcf().add_axes([0.9, 0.1, 0.1, 0.8], frame_on=False)
                box.axis('off')
                textstr = f"CORR\nNNSE: {fluxnet_corr_metrics['NNSE']}\nRMSE: {fluxnet_corr_metrics['RMSE']}\nPBIAS: {fluxnet_corr_metrics['PBIAS']}\nBias: {fluxnet_corr_metrics['BIAS']}\nPearson: {fluxnet_corr_metrics['PEARSON']}"
                box.text(0.2, 0.7, textstr, fontsize=10, verticalalignment='top',color='k')

                misc.makedir(os.path.join(savedir_fig,'daily_climatology'))
                fig.savefig(os.path.join(savedir_fig,'daily_climatology',f'{station}.png'),dpi=300,bbox_inches='tight')
                fig.savefig(os.path.join(savedir_fig,'daily_climatology',f'{station}.svg'),dpi=300,bbox_inches='tight')
                

    gdf_fluxnet_site_info_utm12n_nad83.to_parquet(os.path.join(savedir,f'{frequency}_utm12n_nad83.parquet.gzip'),compression='gzip')
    gdf_fluxnet_site_info_utm12n_nad83.to_csv(os.path.join(savedir,f'{frequency}_utm12n_nad83.csv'))
    gdf_fluxnet_site_info_utm12n_nad83.to_file(os.path.join(savedir,f'{frequency}_utm12n_nad83.parquet.shp'))

            # TODO: Remove this part

            # fig,ax = plt.subplots(figsize=(4,3))
            # df_fluxnet_et_station.plot(ax=ax,color='r',ls='-',label='FLUXNET ET')
            # df_nwm_et_station.plot(ax=ax,color='k',ls='-',label='NWM ET')

            # df_fluxnet_precip_station.plot(ax=ax,color='r',ls='--',label='FLUXNET P')
            # df_nwm_precip_station.plot(ax=ax,color='k',ls='--',label='NWM P')

            # ax.legend()

            # df_nwm_station = emf.add_water_year(pd.DataFrame(df_nwm_station))


            # fig,ax = plt.subplots(figsize=(6,3))
            # df_fluxnet_station.index = df_fluxnet_station.index.shift(-7,freq='H')
            # df_nwm_station.index = df_nwm_station.index.shift(-7,freq='H')
            # df_fluxnet_station.groupby(df_fluxnet_station.index.hour).mean().plot(ax=ax,color='r',label='FLUXNET_GAP')
            # df_nwm_station.groupby(df_nwm_station.index.hour).mean().plot(ax=ax,color='k',label='NWM')

            # fig,ax = plt.subplots(figsize=(6,3))
            # df_fluxnet_corr_station.resample('ME').sum().plot(ax=ax,color='b',label='FLUXNET_CORR')
            # df_fluxnet_station.resample('ME').sum().plot(ax=ax,color='r',label='FLUXNET_GAP')
            # df_nwm_station.resample('ME').sum().plot(ax=ax,color='k',label='NWM')
            # ax.legend()
            # ax.set_xlabel(None)
            # ax.set_ylabel('ET (mm/month)')
            # ax.set_title(f'{station}')
            # plt.tight_layout()
            # fig.savefig(os.path.join(savedir_fig,f'{station}.png'),dpi=300,bbox_inches='tight')

                # df_fluxnet_corr_station = df_fluxnet_corr_station.resample('3H').sum()
                # df_nwm_station = df_nwm_station.resample('3H').sum()

            # test = emf.preprocess_df_metrics(df_fluxnet_station,df_nwm_station)
            # test1 = emf.preprocess_df_metrics(df_fluxnet_corr_station,df_nwm_station)

            # df_fluxnet_station = 

            # fig,ax = plt.subplots(figsize=(4,3))
            # df_fluxnet_corr_station.dropna().resample('A').sum().plot(ax=ax,color='b',label='FLUXNET_CORR')
            # df_fluxnet_station.dropna().resample('A').sum().plot(ax=ax,color='r',label='FLUXNET_GAP')
            # test[1].resample('A').sum().plot(ax=ax,color='k',label='NWM')
            # ax.legend()
            # ax.set_title(f'{station}')


