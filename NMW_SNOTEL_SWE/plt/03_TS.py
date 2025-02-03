import os
import sys
sys.path.append('../src')
import glob

import xarray as xr
import numpy as np
import pandas as pd
import geopandas as gpd

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.dates as mdates
from matplotlib.ticker import NullLocator
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import eval_metrics_snotel as ems
import nwm_static_param_dict as nspd

import param_nwm3
import misc

mpl.rcParams['pdf.fonttype'] = 42

# Output Directory
savedir = os.path.join(param_nwm3.out_dir,'figures',os.path.basename(__file__).split('.')[0])
misc.makedir(savedir)

# Data Directory
snotel_data_dir = '../out/00_download_SNOTEL_data/parquet'
nwm_data_dir = '../out/01_download_NWM_data'
metrics_dir = '../out/02_calc_metrics'

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

frequency = 'daily'
start_wy = 2003
end_wy = 2022
p_thres = 80
min_wy_thres = 15

if frequency == 'hourly':
    data_dict = hourly_data_dict
elif frequency == 'daily':
    data_dict = daily_data_dict
elif frequency == 'monthly':
    data_dict = daily_data_dict

if frequency == 'daily':
    # Daily Plot (SWE)
    gdf_metrics = gpd.read_parquet(os.path.join(metrics_dir,'{}_{}_{}_utm12n_nad83.parquet.gzip'.format(frequency,start_wy,end_wy)))
    
    # # Define the custom order
    # custom_order = ['SJ', 'LCO', 'VE', 'SA', 'UG']

    # # Convert 'M_BASIN_ABR' to a categorical type with the custom order
    # gdf_metrics['M_BASIN_ABR'] = pd.Categorical(gdf_metrics['M_BASIN_ABR'], categories=custom_order, ordered=True)

    # # Sort the DataFrame by 'M_BASIN_ABR'
    # gdf_metrics = gdf_metrics.sort_values('M_BASIN_ABR')

    gdf_metrics = gdf_metrics.sort_values('SNOTEL_Elevation_m',ascending=False)

    df_snotel = pd.read_parquet(data_dict['SWE']['SNOTEL'])
    df_nwm = pd.read_parquet(data_dict['SWE']['NWM'])

    # df_snotel = df_snotel/1000
    # df_nwm = df_nwm/1000

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

    fig,axs = plt.subplots(7,3,figsize=(8,12),sharex=False,sharey=False)
    i=0
    for ax in axs.flatten():
        if (i == 19) | (i == 20):
            ax.axis('off')
        else:
            station = gdf_metrics.iloc[i].name
            basin = gdf_metrics.iloc[i].M_BASIN_ABR
            elevation = gdf_metrics.iloc[i].SNOTEL_Elevation_m

            df_snotel_station = df_snotel.loc[:,station]
            df_nwm_station = df_nwm.loc[:,station]

            dummy_dt_index = pd.date_range(start='2022-01-01',end='2022-12-31',freq='D')
            
            # Calc Mean
            df_snotel_mean = df_snotel_station.groupby(df_snotel_station.index.dayofyear).mean()
            df_nwm_mean = df_nwm_station.groupby(df_nwm_station.index.dayofyear).mean()
            df_mean = pd.concat([df_snotel_mean,df_nwm_mean],axis=1)
            df_mean.columns = ['SNOTEL','NWM']
            df_mean = df_mean.iloc[:-1]
            df_mean.index = dummy_dt_index
            df_mean.index = df_mean.index.map(lambda x: x.replace(year=x.year-1) if x.month >= 10 else x)
            df_mean = df_mean.sort_index()
            df_mean = df_mean.loc[df_mean.index.month.isin([10, 11, 12, 1, 2, 3, 4, 5, 6, 7, 8])]
            


            # Calc 5-95 Percentile
            df_snotel_5 = df_snotel_station.groupby(df_snotel_station.index.dayofyear).quantile(0.05)
            df_nwm_5 = df_nwm_station.groupby(df_nwm_station.index.dayofyear).quantile(0.05)
            df_5 = pd.concat([df_snotel_5,df_nwm_5],axis=1)
            df_5.columns = ['SNOTEL','NWM']
            df_5 = df_5.iloc[:-1]
            df_5.index = dummy_dt_index
            df_5.index = df_5.index.map(lambda x: x.replace(year=x.year-1) if x.month >= 10 else x)
            df_5 = df_5.sort_index()
            df_5 = df_5.loc[df_5.index.month.isin([10, 11, 12, 1, 2, 3, 4, 5, 6, 7, 8])]
            

            df_snotel_95 = df_snotel_station.groupby(df_snotel_station.index.dayofyear).quantile(0.95)
            df_nwm_95 = df_nwm_station.groupby(df_nwm_station.index.dayofyear).quantile(0.95)
            df_95 = pd.concat([df_snotel_95,df_nwm_95],axis=1)
            df_95.columns = ['SNOTEL','NWM']
            df_95 = df_95.iloc[:-1]
            df_95.index = dummy_dt_index
            df_95.index = df_95.index.map(lambda x: x.replace(year=x.year-1) if x.month >= 10 else x)
            df_95 = df_95.sort_index()
            df_95 = df_95.loc[df_95.index.month.isin([10, 11, 12, 1, 2, 3, 4, 5, 6, 7, 8])]
            

            # Plot
            df_mean.loc[:,'SNOTEL'].plot(ax=ax,color='k',ls='-',lw=0.8,label='SNOTEL')
            ax.fill_between(df_mean.index,df_5.loc[:,'SNOTEL'],df_95.loc[:,'SNOTEL'],color='k',alpha=0.2,lw=0.5)

            df_mean.loc[:,'NWM'].plot(ax=ax,color='r',ls='--',lw=0.8,label='NWM')
            ax.fill_between(df_mean.index,df_5.loc[:,'NWM'],df_95.loc[:,'NWM'],color='r',alpha=0.2,lw=0.5)
            ax.set_ylim(bottom=0)
            
            ax.xaxis.set_major_locator(mdates.MonthLocator())
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
            fmt = ax.xaxis.get_major_formatter()
            ax.xaxis.set_major_formatter(lambda x, pos: fmt(x, pos)[0])

            # Remove minor ticks
            ax.xaxis.set_minor_locator(NullLocator())
            ax.yaxis.set_minor_locator(NullLocator())

            if i >= 16:
                ax.set_xticklabels(ax.get_xticklabels(),fontsize=12)
            else:
                ax.set_xticklabels([])
            ax.set_yticklabels(ax.get_yticklabels(),fontsize=12)
            
            ax.grid(True,ls='--',color='gray',alpha=0.5)
            ax.set_title(f'{station} ({basin})',fontsize=12)
            
            NNSE = gdf_metrics.loc[station,'SWE_NNSE']
            PBIAS = gdf_metrics.loc[station,'SWE_PBIAS']
            RMSE = gdf_metrics.loc[station,'SWE_RMSE']



            ax.text(0.98, 0.98, f'NNSE: {NNSE:.2f}\nRMSE: {RMSE:.1f}\nPBIAS: {PBIAS:.1f}', 
                transform=ax.transAxes, verticalalignment='top', fontsize=10,horizontalalignment='right',)
            ax.text(-0.1, 1.17, f'({chr(97 + i)})', transform=ax.transAxes, verticalalignment='top', fontsize=12)
            ax.text(0.02, 0.98, f'{elevation:.1f} m', transform=ax.transAxes, verticalalignment='top', fontsize=10, horizontalalignment='left')

        # Create a legend with a dashed line inside a shaded region
        fluxnet_line = mlines.Line2D([], [], color='k', linestyle='-', linewidth=0.8, label='SNOTEL')
        fluxnet_patch = mpatches.Patch(color='k', alpha=0.2, label='SNOTEL')
        nwm_line = mlines.Line2D([], [], color='r', linestyle='--', linewidth=0.8, label='NWM')
        nwm_patch = mpatches.Patch(color='r', alpha=0.2, label='NWM')

        if i==18:
            # Add a legend for the whole figure below the bottom of the figure
            fig.legend(handles=[fluxnet_line, fluxnet_patch, nwm_line, nwm_patch], loc='lower center', ncol=2, bbox_to_anchor=(0.7, 0.07),prop={'size': 12})


        i+=1
    plt.tight_layout()
    fig.savefig(os.path.join(savedir,'{}_SNOTEL.png'.format(frequency)),dpi=300,bbox_inches='tight')
    fig.savefig(os.path.join(savedir,'{}_SNOTEL.svg'.format(frequency)),dpi=300,bbox_inches='tight')
    fig.savefig(os.path.join(savedir,'{}_SNOTEL.pdf'.format(frequency)),dpi=300,bbox_inches='tight')


if frequency == 'hourly':
    # Hourly Plot (SWE)
    gdf_metrics = gpd.read_parquet(os.path.join(metrics_dir,'{}_{}_{}_utm12n_nad83.parquet.gzip'.format(frequency,start_wy,end_wy)))
    
    # # Define the custom order
    # custom_order = ['SJ', 'LCO', 'VE', 'SA', 'UG']

    # # Convert 'M_BASIN_ABR' to a categorical type with the custom order
    # gdf_metrics['M_BASIN_ABR'] = pd.Categorical(gdf_metrics['M_BASIN_ABR'], categories=custom_order, ordered=True)

    # # Sort the DataFrame by 'M_BASIN_ABR'
    # gdf_metrics = gdf_metrics.sort_values('M_BASIN_ABR')

    gdf_metrics = gdf_metrics.sort_values('SNOTEL_Elevation_m',ascending=False)

    df_snotel = pd.read_parquet(data_dict['SWE']['SNOTEL'])
    df_nwm = pd.read_parquet(data_dict['SWE']['NWM'])

    # df_snotel = df_snotel/1000
    # df_nwm = df_nwm/1000

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

    fig,axs = plt.subplots(7,3,figsize=(8,12),sharex=False,sharey=False)
    i=0
    for ax in axs.flatten():
        if (i == 19) | (i == 20):
            ax.axis('off')
        else:
            station = gdf_metrics.iloc[i].name
            basin = gdf_metrics.iloc[i].M_BASIN_ABR
            elevation = gdf_metrics.iloc[i].SNOTEL_Elevation_m

            df_snotel_station = df_snotel.loc[:,station]
            df_nwm_station = df_nwm.loc[:,station]

            dummy_dt_index = pd.date_range(start='2022-01-01',end='2022-12-31',freq='D')
            
            # Calc Mean
            df_snotel_mean = df_snotel_station.groupby(df_snotel_station.index.dayofyear).mean()
            df_nwm_mean = df_nwm_station.groupby(df_nwm_station.index.dayofyear).mean()
            df_mean = pd.concat([df_snotel_mean,df_nwm_mean],axis=1)
            df_mean.columns = ['SNOTEL','NWM']
            df_mean = df_mean.iloc[:-1]
            df_mean.index = dummy_dt_index
            df_mean.index = df_mean.index.map(lambda x: x.replace(year=x.year-1) if x.month >= 10 else x)
            df_mean = df_mean.sort_index()
            df_mean = df_mean.loc[df_mean.index.month.isin([10, 11, 12, 1, 2, 3, 4, 5, 6, 7, 8])]
            


            # Calc 5-95 Percentile
            df_snotel_5 = df_snotel_station.groupby(df_snotel_station.index.dayofyear).quantile(0.05)
            df_nwm_5 = df_nwm_station.groupby(df_nwm_station.index.dayofyear).quantile(0.05)
            df_5 = pd.concat([df_snotel_5,df_nwm_5],axis=1)
            df_5.columns = ['SNOTEL','NWM']
            df_5 = df_5.iloc[:-1]
            df_5.index = dummy_dt_index
            df_5.index = df_5.index.map(lambda x: x.replace(year=x.year-1) if x.month >= 10 else x)
            df_5 = df_5.sort_index()
            df_5 = df_5.loc[df_5.index.month.isin([10, 11, 12, 1, 2, 3, 4, 5, 6, 7, 8])]
            

            df_snotel_95 = df_snotel_station.groupby(df_snotel_station.index.dayofyear).quantile(0.95)
            df_nwm_95 = df_nwm_station.groupby(df_nwm_station.index.dayofyear).quantile(0.95)
            df_95 = pd.concat([df_snotel_95,df_nwm_95],axis=1)
            df_95.columns = ['SNOTEL','NWM']
            df_95 = df_95.iloc[:-1]
            df_95.index = dummy_dt_index
            df_95.index = df_95.index.map(lambda x: x.replace(year=x.year-1) if x.month >= 10 else x)
            df_95 = df_95.sort_index()
            df_95 = df_95.loc[df_95.index.month.isin([10, 11, 12, 1, 2, 3, 4, 5, 6, 7, 8])]
            

            # Plot
            df_mean.loc[:,'SNOTEL'].plot(ax=ax,color='k',ls='-',lw=0.8,label='SNOTEL')
            ax.fill_between(df_mean.index,df_5.loc[:,'SNOTEL'],df_95.loc[:,'SNOTEL'],color='k',alpha=0.2,lw=0.5)

            df_mean.loc[:,'NWM'].plot(ax=ax,color='r',ls='--',lw=0.8,label='NWM')
            ax.fill_between(df_mean.index,df_5.loc[:,'NWM'],df_95.loc[:,'NWM'],color='r',alpha=0.2,lw=0.5)
            ax.set_ylim(bottom=0)
            
            ax.xaxis.set_major_locator(mdates.MonthLocator())
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
            fmt = ax.xaxis.get_major_formatter()
            ax.xaxis.set_major_formatter(lambda x, pos: fmt(x, pos)[0])

            # Remove minor ticks
            ax.xaxis.set_minor_locator(NullLocator())
            ax.yaxis.set_minor_locator(NullLocator())

            if i >= 16:
                ax.set_xticklabels(ax.get_xticklabels(),fontsize=12)
            else:
                ax.set_xticklabels([])
            ax.set_yticklabels(ax.get_yticklabels(),fontsize=12)
            
            ax.grid(True,ls='--',color='gray',alpha=0.5)
            ax.set_title(f'{station} ({basin})',fontsize=12)
            
            NNSE = gdf_metrics.loc[station,'SWE_NNSE']
            PBIAS = gdf_metrics.loc[station,'SWE_PBIAS']
            RMSE = gdf_metrics.loc[station,'SWE_RMSE']



            ax.text(0.98, 0.98, f'NNSE: {NNSE:.2f}\nRMSE: {RMSE:.1f}\nPBIAS: {PBIAS:.1f}', 
                transform=ax.transAxes, verticalalignment='top', fontsize=10,horizontalalignment='right',)
            ax.text(-0.1, 1.17, f'({chr(97 + i)})', transform=ax.transAxes, verticalalignment='top', fontsize=12)
            ax.text(0.02, 0.98, f'{elevation:.1f} m', transform=ax.transAxes, verticalalignment='top', fontsize=10, horizontalalignment='left')



        # Create a legend with a dashed line inside a shaded region
        fluxnet_line = mlines.Line2D([], [], color='k', linestyle='-', linewidth=0.8, label='SNOTEL')
        fluxnet_patch = mpatches.Patch(color='k', alpha=0.2, label='SNOTEL')
        nwm_line = mlines.Line2D([], [], color='r', linestyle='--', linewidth=0.8, label='NWM')
        nwm_patch = mpatches.Patch(color='r', alpha=0.2, label='NWM')

        if i==18:
            # Add a legend for the whole figure below the bottom of the figure
            fig.legend(handles=[fluxnet_line, fluxnet_patch, nwm_line, nwm_patch], loc='lower center', ncol=2, bbox_to_anchor=(0.7, 0.07),prop={'size': 12})



        i+=1

    plt.tight_layout()
    fig.savefig(os.path.join(savedir,'{}_SNOTEL.png'.format(frequency)),dpi=300,bbox_inches='tight')
    fig.savefig(os.path.join(savedir,'{}_SNOTEL.svg'.format(frequency)),dpi=300,bbox_inches='tight')
    fig.savefig(os.path.join(savedir,'{}_SNOTEL.pdf'.format(frequency)),dpi=300,bbox_inches='tight')

    