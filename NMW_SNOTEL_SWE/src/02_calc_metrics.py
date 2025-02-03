import os

import xarray as xr
import numpy as np
import pandas as pd
import geopandas as gpd
import datetime

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import FuncFormatter

import param_nwm3
import misc
import eval_metrics_snotel as ems
import nwm_static_param_dict as nspd

def m_fmt(x, pos=None):
    month_fmt = mdates.DateFormatter('%b')
    return month_fmt(x)[0]

# Output Directory
savedir = os.path.join(param_nwm3.out_dir,os.path.basename(__file__).split('.')[0])
savedir_fig = os.path.join(param_nwm3.out_dir,'figures',os.path.basename(__file__).split('.')[0])
misc.makedir(savedir)
misc.makedir(savedir_fig)

# SNOTEL Site Info
gp_snotel_site_info_utm12n_nad83 = '../out/00_download_SNOTEL_data/info/snotel_site_info_utm12n_nad83.parquet.gzip'
gdf_snotel_site_info_utm12n_nad83 = gpd.read_file(gp_snotel_site_info_utm12n_nad83)
gdf_snotel_site_info_nwm_lcc = gdf_snotel_site_info_utm12n_nad83.to_crs(param_nwm3.crs_nwm_proj4_lcc)

# NWM Static Properties
ds_nwm_hgt_az_buffered_utm12n_nad83 = xr.open_dataset(param_nwm3.nc_nwm_hgt_az_buffered_utm12n_nad83)['HGT']
ds_nwm_isltyp_az_buffered_utm12n_nad83 = xr.open_dataset(param_nwm3.nc_nwm_isltyp_az_buffered_utm12n_nad83)['ISLTYP']
ds_nwm_ivgtyp_az_buffered_utm12n_nad83 = xr.open_dataset(param_nwm3.nc_nwm_ivgtyp_az_buffered_utm12n_nad83)['IVGTYP']

# Merged Basins
gdf_m_basins_az_utm12n_nad83 = gpd.read_file(param_nwm3.shp_az_simplified_watersheds_utm12n_nad83)

# Data Directory
snotel_data_dir = '../out/00_download_SNOTEL_data/parquet'
nwm_data_dir = '../out/01_download_NWM_data'

daily_data_dict = {'SWE':{'SNOTEL':os.path.join(snotel_data_dir,'WTEQ_D.parquet.gzip'),
                          'NWM':os.path.join(nwm_data_dir,'SNEQV_D.parquet.gzip')},
                   'SNOWD':{'SNOTEL':os.path.join(snotel_data_dir,'SNWD_D.parquet.gzip'),
                           'NWM':os.path.join(nwm_data_dir,'SNOWH_D.parquet.gzip')},
                    'PRECIP':{'SNOTEL':os.path.join(snotel_data_dir,'PRCPSA_D.parquet.gzip'),
                              'NWM':os.path.join(nwm_data_dir,'PRECIP_D.parquet.gzip')},
                    'T2D':{'SNOTEL':os.path.join(snotel_data_dir,'TOBS_D.parquet.gzip'),
                            'NWM':os.path.join(nwm_data_dir,'T2D_D.parquet.gzip')}}

hourly_data_dict = {'SWE':{'SNOTEL':os.path.join(snotel_data_dir,'WTEQ_H.parquet.gzip'),
                            'NWM':os.path.join(nwm_data_dir,'SNEQV_3H.parquet.gzip')},
                    'SNOWD':{'SNOTEL':os.path.join(snotel_data_dir,'SNWD_H.parquet.gzip'),
                             'NWM':os.path.join(nwm_data_dir,'SNOWH_3H.parquet.gzip')},
                    'T2D':{'SNOTEL':os.path.join(snotel_data_dir,'TOBS_H.parquet.gzip'),
                            'NWM':os.path.join(nwm_data_dir,'T2D_H.parquet.gzip')}}


frequencies = [
    'hourly',
    'daily',
    'monthly'
]

sel_vars = [
    'SWE',
    'SNOWD',
    # 'PRECIP',
    # 'T2D'
]

start_wy = 2003
end_wy = 2022
p_thres = 80
min_wy_thres = 15

# Write Static Properties
gdf_snotel_site_info_utm12n_nad83 = gpd.sjoin(gdf_snotel_site_info_utm12n_nad83,gdf_m_basins_az_utm12n_nad83,how='left',predicate='within')
gdf_snotel_site_info_utm12n_nad83 = gdf_snotel_site_info_utm12n_nad83.rename(columns={'NAME':'M_BASIN',
                                                                                      'NAME_ABR':'M_BASIN_ABR'})
gdf_snotel_site_info_utm12n_nad83 = gdf_snotel_site_info_utm12n_nad83.drop(columns=['index_right','area_m2','__index_level_0__'])
gdf_snotel_site_info_utm12n_nad83.index = gdf_snotel_site_info_utm12n_nad83['code']


# Write Static NWM Properties
gdfs_static_data = [ds_nwm_hgt_az_buffered_utm12n_nad83,
                    ds_nwm_isltyp_az_buffered_utm12n_nad83,
                    ds_nwm_ivgtyp_az_buffered_utm12n_nad83]
dfs_static_data = []
for gdf_static_data in gdfs_static_data:
    var_name = gdf_static_data.name
    df_static_data = ems.get_static_nwm_data(gdf_static_data,gdf_snotel_site_info_utm12n_nad83,var_name)
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

gdf_snotel_site_info_utm12n_nad83 = pd.concat([gdf_snotel_site_info_utm12n_nad83,dfs_static_data],axis=1)
gdf_snotel_site_info_utm12n_nad83 = gdf_snotel_site_info_utm12n_nad83.rename(columns={'elevation_m':'SNOTEL_Elevation_m'})
gdf_snotel_site_info_utm12n_nad83['Delta_Elevation_m'] = gdf_snotel_site_info_utm12n_nad83['NWM_Elevation_m'] - gdf_snotel_site_info_utm12n_nad83['SNOTEL_Elevation_m']

# Calculate Metrics
for frequency in frequencies:
    if frequency == 'hourly':
        data_dict = hourly_data_dict
    elif frequency == 'daily':
        data_dict = daily_data_dict
    elif frequency == 'monthly':
        data_dict = daily_data_dict

    gdf_snotel_site_metrics_utm12n_nad83 = gdf_snotel_site_info_utm12n_nad83.copy()
    for variable in sel_vars:
        print(frequency,variable)
        # Read Data
        df_snotel = pd.read_parquet(data_dict[variable]['SNOTEL'])
        df_nwm = pd.read_parquet(data_dict[variable]['NWM'])

        if frequency == 'hourly':
            df_snotel = df_snotel.resample('3h').mean()
            df_nwm = df_nwm.resample('3h').mean()

        # if variable == 'SWE':
        #     df_snotel = df_snotel/1000
        #     df_nwm = df_nwm/1000
        # elif variable == 'SNOWD':
        #     df_snotel = df_snotel/1000
        #     df_nwm = df_nwm/1000

        # Preprocess Data
        df_snotel = ems.process_df(df_snotel,start_wy,end_wy)
        df_snotel_avail_n_stations = ems.data_availability(df_snotel,p_thres,min_wy_thres)['avail_wy_thres']
        df_snotel = ems.filter_by_avail(df_snotel,p_thres,min_wy_thres)
        df_snotel_avail_n_stations = df_snotel_avail_n_stations[df_snotel.columns]

        df_nwm = ems.process_df(df_nwm,start_wy,end_wy)

        # Select Common Stations
        common_stations = list(set(df_snotel.columns) & set(df_nwm.columns))
        df_snotel = df_snotel[common_stations]
        df_nwm = df_nwm[common_stations]




        if frequency == 'monthly':
            df_snotel = df_snotel.resample('ME').mean()
            df_nwm = df_nwm.resample('ME').mean()
        # Calculate Metrics
        df_obs = df_snotel
        df_sim = df_nwm

        # df_obs = df_obs[df_obs.index.month.isin([10, 11, 12, 1, 2, 3, 4, 5, 6])]
        # df_sim = df_sim[df_sim.index.month.isin([10, 11, 12, 1, 2, 3, 4, 5, 6])]

        nse=ems.NSE(df_obs,df_sim)
        nnse=ems.NNSE(df_obs,df_sim)
        rmse=ems.RMSE(df_obs,df_sim)
        pbias=ems.PBIAS(df_obs,df_sim)
        bias=ems.BIAS(df_obs,df_sim)

        # Save Metrics
        gdf_snotel_site_metrics_utm12n_nad83[f'{variable}_NSE'] = nse
        gdf_snotel_site_metrics_utm12n_nad83[f'{variable}_NNSE'] = nnse
        gdf_snotel_site_metrics_utm12n_nad83[f'{variable}_RMSE'] = rmse
        gdf_snotel_site_metrics_utm12n_nad83[f'{variable}_PBIAS'] = pbias
        gdf_snotel_site_metrics_utm12n_nad83[f'{variable}_BIAS'] = bias


        # Save n_stations
        gdf_snotel_site_metrics_utm12n_nad83[f'{variable}_N_WY'] = df_snotel_avail_n_stations


    # Calculate Mean Annual Precipitation
    df_snotel_precip = pd.read_parquet(daily_data_dict['PRECIP']['SNOTEL'])
    df_nwm_precip = pd.read_parquet(daily_data_dict['PRECIP']['NWM'])
    df_snotel_precip = df_snotel_precip[common_stations]
    df_nwm_precip = df_nwm_precip[common_stations]
    df_snotel_precip = ems.process_df(df_snotel_precip,start_wy,end_wy)
    df_nwm_precip = ems.process_df(df_nwm_precip,start_wy,end_wy)
    # df_snotel_precip = df_snotel_precip.resample('YE').sum()
    # df_nwm_precip = df_nwm_precip.resample('YE').sum()
    df_snotel_precip = ems.groupby_wy_sum(df_snotel_precip)
    df_nwm_precip = ems.groupby_wy_sum(df_nwm_precip)

    # Calculate Mean Annual Temperature
    df_snotel_t2d = pd.read_parquet(daily_data_dict['T2D']['SNOTEL'])
    df_nwm_t2d = pd.read_parquet(daily_data_dict['T2D']['NWM'])
    df_snotel_t2d = df_snotel_t2d[common_stations]
    df_nwm_t2d = df_nwm_t2d[common_stations]
    df_snotel_t2d = ems.process_df(df_snotel_t2d,start_wy,end_wy)
    df_nwm_t2d = ems.process_df(df_nwm_t2d,start_wy,end_wy)
    # df_snotel_t2d = df_snotel_t2d.resample('YE').mean()
    # df_nwm_t2d = df_nwm_t2d.resample('YE').mean()
    df_snotel_t2d = ems.groupby_wy_mean(df_snotel_t2d)
    df_nwm_t2d = ems.groupby_wy_mean(df_nwm_t2d)

    # Calculate Mean Annual Snow Precipitation
    df_snotel_precip = pd.read_parquet(daily_data_dict['PRECIP']['SNOTEL'])
    df_nwm_precip = pd.read_parquet(daily_data_dict['PRECIP']['NWM'])
    df_snotel_precip = df_snotel_precip[common_stations]
    df_nwm_precip = df_nwm_precip[common_stations]
    df_snotel_precip = ems.process_df(df_snotel_precip,start_wy,end_wy)
    df_nwm_precip = ems.process_df(df_nwm_precip,start_wy,end_wy)

    df_snotel_t2d = pd.read_parquet(daily_data_dict['T2D']['SNOTEL'])
    df_nwm_t2d = pd.read_parquet(daily_data_dict['T2D']['NWM'])
    df_snotel_t2d = df_snotel_t2d[common_stations]
    df_nwm_t2d = df_nwm_t2d[common_stations]
    df_snotel_t2d = ems.process_df(df_snotel_t2d,start_wy,end_wy)
    df_nwm_t2d = ems.process_df(df_nwm_t2d,start_wy,end_wy)

    fice_threshold = 273.15
    df_snotel_fice = (df_snotel_t2d.drop(columns='WY')+273.15).applymap(lambda tsfc: ems.fice_jordan1991(tsfc, fice_threshold))
    df_nwm_fice = (df_nwm_t2d.drop(columns='WY')+273.15).applymap(lambda tsfc: ems.fice_jordan1991(tsfc, fice_threshold))

    # Rainfall
    df_snotel_precip_rain = df_snotel_precip*(1-df_snotel_fice)
    df_nwm_precip_rain = df_nwm_precip*(1-df_nwm_fice)
    df_snotel_precip_rain = ems.process_df(df_snotel_precip_rain,start_wy,end_wy)
    df_nwm_precip_rain = ems.process_df(df_nwm_precip_rain,start_wy,end_wy)
    df_snotel_precip_rain = ems.groupby_wy_sum(df_snotel_precip_rain)
    df_nwm_precip_rain = ems.groupby_wy_sum(df_nwm_precip_rain)

    # Snowfall
    df_snotel_precip_snow = df_snotel_precip*df_snotel_fice
    df_nwm_precip_snow = df_nwm_precip*df_nwm_fice
    df_snotel_precip_snow = ems.process_df(df_snotel_precip_snow,start_wy,end_wy)
    df_nwm_precip_snow = ems.process_df(df_nwm_precip_snow,start_wy,end_wy)
    df_snotel_precip_snow = ems.groupby_wy_sum(df_snotel_precip_snow)
    df_nwm_precip_snow = ems.groupby_wy_sum(df_nwm_precip_snow)


    # Save Climatology
    gdf_snotel_site_metrics_utm12n_nad83[f'SNOTEL_PRECIP_WY'] = df_snotel_precip.mean()
    gdf_snotel_site_metrics_utm12n_nad83[f'NWM_PRECIP_WY'] = df_nwm_precip.mean()
    gdf_snotel_site_metrics_utm12n_nad83[f'Delta_PRECIP_WY'] = df_nwm_precip.mean() - df_snotel_precip.mean()
    gdf_snotel_site_metrics_utm12n_nad83[f'SNOTEL_T2D_WY'] = df_snotel_t2d.mean()
    gdf_snotel_site_metrics_utm12n_nad83[f'NWM_T2D_WY'] = df_nwm_t2d.mean()
    gdf_snotel_site_metrics_utm12n_nad83[f'Delta_T2D_WY'] = df_nwm_t2d.mean() - df_snotel_t2d.mean()
    gdf_snotel_site_metrics_utm12n_nad83[f'SNOTEL_RAINFALL_WY'] = df_snotel_precip_rain.mean()
    gdf_snotel_site_metrics_utm12n_nad83[f'NWM_RAINFALL_WY'] = df_nwm_precip_rain.mean()
    gdf_snotel_site_metrics_utm12n_nad83[f'Delta_RAINFALL_WY'] = df_nwm_precip_rain.mean() - df_snotel_precip_rain.mean()
    gdf_snotel_site_metrics_utm12n_nad83[f'SNOTEL_SNOWFALL_WY'] = df_snotel_precip_snow.mean()
    gdf_snotel_site_metrics_utm12n_nad83[f'NWM_SNOWFALL_WY'] = df_nwm_precip_snow.mean()
    gdf_snotel_site_metrics_utm12n_nad83[f'Delta_SNOWFALL_WY'] = df_nwm_precip_snow.mean() - df_snotel_precip_snow.mean()


        # # 01 - Plot SWE Climatology
        # if frequency == 'daily':
        #     if variable in ['SWE','SNOWD']:
        #         fig,axs = plt.subplots(5,4,figsize=(16,16))
        #         i=0
        #         for site in df_snotel.columns:
        #             df_obs_site = df_obs[site]
        #             df_sim_site = df_sim[site]

        #             df_obs_site = ems.cal_snow_plot_metrics(df_obs_site)
        #             df_sim_site = ems.cal_snow_plot_metrics(df_sim_site)

        #             # plt_start_date = datetime.datetime(2000,10,1)
        #             # plt_end_date = datetime.datetime(2001,7,31)

        #             # df_obs_site = df_obs_site.loc[plt_start_date:plt_end_date]
        #             # df_sim_site = df_sim_site.loc[plt_start_date:plt_end_date]

        #             ax = axs.reshape(-1)[i]
        #             # Plot Observed
        #             ax.fill_between(df_obs_site.index,df_obs_site['P5'],df_obs_site['P95'],color='gray',alpha=0.3,label='SNOTEL (5-95%)')
        #             ax.plot(df_obs_site.index,df_obs_site['mean'],label='SNOTEL',color='k',ls='-',lw=1)

        #             # Plot Simulated
        #             ax.fill_between(df_sim_site.index,df_sim_site['P5'],df_sim_site['P95'],color='r',alpha=0.3,label='NWM (5-95%)')
        #             ax.plot(df_sim_site.index,df_sim_site['mean'],label='NWM',color='r',ls='-',lw=1)

        #             ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
        #             ax.xaxis.set_major_formatter(FuncFormatter(m_fmt))
        #             ax.tick_params(axis='both', which='major', labelsize=16)
        #             ax.margins(x=0)
        #             ax.set_ylim(bottom=0)
        #             text_elevation = gdf_snotel_site_info_utm12n_nad83.loc[site,'SNOTEL_Elevation_m']
        #             ax.set_title(site+' ('+str(round(text_elevation,1))+' m)',fontsize=16)

        #             if i%4==0:
        #                 if variable == 'SWE':
        #                     ax.set_ylabel('SWE [m]',fontsize=16)
        #                 elif variable == 'SNOWD':
        #                     ax.set_ylabel('Snow Depth [m]',fontsize=16)

        #             text_str = f'NSE: {nse[site]:.2f}\nNNSE: {nnse[site]:.2f}\nRMSE: {rmse[site]:.2f} m\nPBIAS: {pbias[site]:.1f} %'
        #             ax.text(0.50,0.95,text_str,transform=ax.transAxes,fontsize=14,va='top',ha='left',bbox=dict(facecolor='None',edgecolor='None'))

        #             ax.set_axisbelow(True)
        #             ax.xaxis.grid(color='gray', linestyle='dashed')
        #             ax.yaxis.grid(color='gray', linestyle='dashed')
        #             i+=1
        #         axs.reshape(-1)[-1].remove()
        #         plt.tight_layout()
        #         ax.legend(loc='upper center',bbox_to_anchor=(1.6,0.8),ncol=1,fontsize=16,frameon=False)
        #         fig.savefig(os.path.join(savedir_fig,f'{variable}_{frequency}_metrics.png'),dpi=300,bbox_inches='tight')
        #         fig.savefig(os.path.join(savedir_fig,f'{variable}_{frequency}_metrics.pdf'),dpi=300,bbox_inches='tight')


        # # 02 - Plot SWE Climatology (BIAS)
        # if frequency == 'daily':
        #     if variable in ['SWE','SNOWD']:
        #         fig,axs = plt.subplots(5,4,figsize=(16,16),sharey=True)
        #         i=0
        #         for site in df_snotel.columns:
        #             df_obs_site = df_obs[site]
        #             df_sim_site = df_sim[site]

        #             df_bias = ems.BIAS_metrics(df_obs_site*1000,df_sim_site*1000)


        #             # plt_start_date = datetime.datetime(2000,10,1)
        #             # plt_end_date = datetime.datetime(2001,7,31)

        #             # df_obs_site = df_obs_site.loc[plt_start_date:plt_end_date]
        #             # df_sim_site = df_sim_site.loc[plt_start_date:plt_end_date]

        #             ax = axs.reshape(-1)[i]
        #             # # Plot Observed
        #             # ax.fill_between(df_obs_site.index,df_obs_site['P5'],df_obs_site['P95'],color='gray',alpha=0.3,label='SNOTEL (5-95%)')
        #             # ax.plot(df_obs_site.index,df_obs_site['mean'],label='SNOTEL',color='k',ls='-',lw=1)

        #             # Plot Simulated
        #             ax.fill_between(df_bias.index,df_bias['P5'],df_bias['P95'],color='r',alpha=0.3,label='90% CI')
        #             ax.plot(df_bias.index,df_bias['mean'],label='Mean',color='r',ls='-',lw=1)

        #             ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
        #             ax.xaxis.set_major_formatter(FuncFormatter(m_fmt))
        #             ax.tick_params(axis='both', which='major', labelsize=16)
        #             ax.margins(x=0)
        #             # ax.set_ylim(bottom=0)
        #             text_elevation = gdf_snotel_site_info_utm12n_nad83.loc[site,'SNOTEL_Elevation_m']
        #             ax.set_title(site+' ('+str(round(text_elevation,1))+' m)',fontsize=16)

        #             if i%4==0:
        #                 if variable == 'SWE':
        #                     ax.set_ylabel('SWE BIAS [mm]',fontsize=16)
        #                 elif variable == 'SNOWD':
        #                     ax.set_ylabel('Snow Depth BIAS [mm]',fontsize=16)

        #             text_str = f'NSE: {nse[site]:.2f}\nNNSE: {nnse[site]:.2f}\nRMSE: {rmse[site]:.2f} m\nBIAS: {bias[site]*1000:.1f} mm'
        #             ax.text(0.48,0.4,text_str,transform=ax.transAxes,fontsize=14,va='top',ha='left',bbox=dict(facecolor='None',edgecolor='None'))

        #             ax.set_axisbelow(True)
        #             ax.xaxis.grid(color='gray', linestyle='dashed')
        #             ax.yaxis.grid(color='gray', linestyle='dashed')
        #             i+=1
        #         axs.reshape(-1)[-1].remove()
        #         plt.tight_layout()
        #         ax.legend(loc='upper center',bbox_to_anchor=(1.6,0.8),ncol=1,fontsize=16,frameon=False)
        #         fig.savefig(os.path.join(savedir_fig,f'{variable}_BIAS_{frequency}_metrics.png'),dpi=300,bbox_inches='tight')
        #         fig.savefig(os.path.join(savedir_fig,f'{variable}_BIAS_{frequency}_metrics.pdf'),dpi=300,bbox_inches='tight')
        
        
    # Save Metrics
    gdf_snotel_site_metrics_utm12n_nad83 = gdf_snotel_site_metrics_utm12n_nad83[gdf_snotel_site_metrics_utm12n_nad83['SWE_NNSE'].notnull()]
    gdf_snotel_site_metrics_utm12n_nad83.to_parquet(os.path.join(savedir,f'{frequency}_{str(start_wy)}_{str(end_wy)}_utm12n_nad83.parquet.gzip'),compression='gzip')
    gdf_snotel_site_metrics_utm12n_nad83.to_csv(os.path.join(savedir,f'{frequency}_{str(start_wy)}_{str(end_wy)}_utm12n_nad83.csv'))
    gdf_snotel_site_metrics_utm12n_nad83.to_file(os.path.join(savedir,f'{frequency}_{str(start_wy)}_{str(end_wy)}_utm12n_nad83.shp'))
