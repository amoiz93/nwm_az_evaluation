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
import eval_metrics_fluxnet as emf


import param_nwm3
import misc

mpl.rcParams['pdf.fonttype'] = 42

# Output Directory
savedir = os.path.join(param_nwm3.out_dir,'figures',os.path.basename(__file__).split('.')[0])
misc.makedir(savedir)

# Data Directory
fluxnet_data_dir = '../out/00_proc_FLUXNET_data/parquet'
nwm_data_dir = '../out/01_download_NWM_ET'
metrics_dir = '../out/02_calc_metrics'

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

frequency = 'daily'

# Define the order of the categories
igbp_order = ['ENF','GRA','OSH','SAV','WSA']


if frequency == 'hourly':
    data_dict = hourly_data_dict
elif frequency == 'daily':
    data_dict = daily_data_dict
elif frequency == 'monthly':
    data_dict = monthly_data_dict



if frequency == 'daily':
    # Daily plot
    gdf_metrics = gpd.read_parquet(os.path.join(metrics_dir,'{}_utm12n_nad83.parquet.gzip'.format(frequency)))
    df_fluxnet = pd.read_parquet(data_dict['ET']['FLUXNET_GAP'])
    df_fluxnet_corr = pd.read_parquet(data_dict['ET']['FLUXNET_CORR'])
    df_nwm = pd.read_parquet(data_dict['ET']['NWM'])

    # df_fluxnet_corr = df_fluxnet_corr.rolling(window=7).mean()
    # df_nwm = df_nwm.rolling(window=7).mean()


    # Convert the 'igbp' column to a categorical data type
    gdf_metrics['igbp'] = pd.Categorical(gdf_metrics['igbp'], categories=igbp_order, ordered=True)

    # Sort the DataFrame by 'igbp'
    gdf_metrics = gdf_metrics.sort_values('igbp')


    fig, axs = plt.subplots(3,3,figsize=(8,8),sharex=True,sharey=True)
    i=0
    for ax in axs.flatten():
        station = gdf_metrics.iloc[i].name
        igbp = gdf_metrics.loc[station,'igbp']

        start_wy,end_wy,n_wy = emf.data_availabilty(df_fluxnet[[station]])

        df_fluxnet_station = df_fluxnet.loc[:,station]
        df_fluxnet_corr_station = df_fluxnet_corr.loc[:,station]
        df_nwm_station = df_nwm.loc[:,station]

        

        df_fluxnet_station = df_fluxnet_station.dropna()
        df_fluxnet_station = df_fluxnet_station.resample('D').sum()

        df_fluxnet_corr_station = df_fluxnet_corr_station.dropna()
        df_fluxnet_corr_station = df_fluxnet_corr_station.resample('D').sum()

        df_nwm_station = df_nwm_station.dropna()
        df_nwm_station = df_nwm_station.resample('D').sum()

        df_nwm_station = df_nwm_station.loc[df_fluxnet_station.index]
        # Convert to US/Arizona Timezone
        df_fluxnet_station.index = df_fluxnet_station.index.tz_localize('UTC').tz_convert('US/Arizona')
        df_fluxnet_corr_station.index = df_fluxnet_corr_station.index.tz_localize('UTC').tz_convert('US/Arizona')
        df_nwm_station.index = df_nwm_station.index.tz_localize('UTC').tz_convert('US/Arizona')


        dummy_dt_index = pd.date_range(start='2020-01-01',end='2020-12-31',freq='D')
        # Calc Mean
        df_fluxnet_corr_mean = df_fluxnet_corr_station.groupby(df_fluxnet_corr_station.index.dayofyear).mean()
        df_nwm_mean = df_nwm_station.groupby(df_nwm_station.index.dayofyear).mean()
        df_mean = pd.concat([df_fluxnet_corr_mean,df_nwm_mean],axis=1)
        df_mean.columns = ['FLUXNET','NWM']
        df_mean.index = dummy_dt_index
        df_mean.index = df_mean.index.map(lambda x: x.replace(year=x.year-1) if x.month >= 10 else x)
        df_mean = df_mean.sort_index()

        # Calc Std
        df_fluxnet_corr_std = df_fluxnet_corr_station.groupby(df_fluxnet_corr_station.index.dayofyear).std()
        df_nwm_std = df_nwm_station.groupby(df_nwm_station.index.dayofyear).std()
        df_std = pd.concat([df_fluxnet_corr_std,df_nwm_std],axis=1)
        df_std.columns = ['FLUXNET','NWM']
        df_std.index = dummy_dt_index
        df_std.index = df_std.index.map(lambda x: x.replace(year=x.year-1) if x.month >= 10 else x)
        df_std = df_std.sort_index()

        # Calc 5-95 Percentile
        df_fluxnet_corr_5 = df_fluxnet_corr_station.groupby(df_fluxnet_corr_station.index.dayofyear).quantile(0.05)
        df_nwm_5 = df_nwm_station.groupby(df_nwm_station.index.dayofyear).quantile(0.05)
        df_5 = pd.concat([df_fluxnet_corr_5,df_nwm_5],axis=1)
        df_5.columns = ['FLUXNET','NWM']
        df_5.index = dummy_dt_index
        df_5.index = df_5.index.map(lambda x: x.replace(year=x.year-1) if x.month >= 10 else x)
        df_5 = df_5.sort_index()

        df_fluxnet_corr_95 = df_fluxnet_corr_station.groupby(df_fluxnet_corr_station.index.dayofyear).quantile(0.95)
        df_nwm_95 = df_nwm_station.groupby(df_nwm_station.index.dayofyear).quantile(0.95)
        df_95 = pd.concat([df_fluxnet_corr_95,df_nwm_95],axis=1)
        df_95.columns = ['FLUXNET','NWM']
        df_95.index = dummy_dt_index
        df_95.index = df_95.index.map(lambda x: x.replace(year=x.year-1) if x.month >= 10 else x)
        df_95 = df_95.sort_index()

        # Plot
        df_mean.loc[:,'FLUXNET'].plot(ax=ax,color='k',ls='-',lw=0.8,label='FLUXNET')
        # ax.fill_between(df_mean.index,df_mean['FLUXNET']-df_std['FLUXNET'],df_mean['FLUXNET']+df_std['FLUXNET'],color='k',alpha=0.2)
        ax.fill_between(df_mean.index,df_5['FLUXNET'],df_95['FLUXNET'],color='k',alpha=0.2,lw=0.5)

        df_mean.loc[:,'NWM'].plot(ax=ax,color='r',ls='--',lw=0.8,label='NWM')
        # ax.fill_between(df_mean.index,df_mean['NWM']-df_std['NWM'],df_mean['NWM']+df_std['NWM'],color='r',alpha=0.2)
        ax.fill_between(df_mean.index,df_5['NWM'],df_95['NWM'],color='r',alpha=0.2,lw=0.5)

        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
        fmt = ax.xaxis.get_major_formatter()
        ax.xaxis.set_major_formatter(lambda x, pos: fmt(x, pos)[0])

        # Remove minor ticks
        ax.xaxis.set_minor_locator(NullLocator())
        ax.yaxis.set_minor_locator(NullLocator())
        
        ax.grid(True,ls='--',color='gray',alpha=0.5)
        # Rotate the x-axis labels
        # plt.xticks(rotation=90)
        ax.set_title(station.replace('US-', '')+' ({})'.format(igbp),fontsize=14)
        ax.tick_params(axis='both', which='major', labelsize=12)
        ax.set_ylabel('$ET$ (mm/day)',fontsize=12)
        # ax.axis('off')

        NNSE = gdf_metrics.loc[station,'ET_NNSE_C_WY']
        PBIAS = gdf_metrics.loc[station,'ET_PBIAS_C_WY']
        RMSE = gdf_metrics.loc[station,'ET_RMSE_C_WY']
        CC = gdf_metrics.loc[station,'ET_PEARSON_C_WY']

        # Show NNSE, PBIAS, RMSE, and CC in the top left corner of the subplot
        # ax.text(0.02, 0.98, f'RMSE: {RMSE:.1f}\nNNSE: {NNSE:.2f}\nPBIAS: {PBIAS:.1f}\n$r$: {CC:.2f}', 
        #         transform=ax.transAxes, verticalalignment='top', fontsize=12)
        ax.text(0.02, 0.98, f'RMSE: {RMSE:.2f}\nNNSE: {NNSE:.2f}\nPBIAS: {PBIAS:.1f}', 
                transform=ax.transAxes, verticalalignment='top', fontsize=12)
        ax.text(0.01, 1.11, f'({chr(97 + i)})', transform=ax.transAxes, verticalalignment='top', fontsize=14)
        ax.text(0.98, 0.98, f'{start_wy}-{end_wy}', 
                transform=ax.transAxes, horizontalalignment='right', verticalalignment='top', fontsize=12)


        # Create a legend with a dashed line inside a shaded region
        fluxnet_line = mlines.Line2D([], [], color='k', linestyle='-', linewidth=0.8, label='FLUXNET')
        fluxnet_patch = mpatches.Patch(color='k', alpha=0.2, label='FLUXNET')
        nwm_line = mlines.Line2D([], [], color='r', linestyle='--', linewidth=0.8, label='NWM')
        nwm_patch = mpatches.Patch(color='r', alpha=0.2, label='NWM')

        if i==0:
            # Add a legend for the whole figure below the bottom of the figure
            fig.legend(handles=[fluxnet_line, fluxnet_patch, nwm_line, nwm_patch], loc='lower center', ncol=2, bbox_to_anchor=(0.5, -0.07),prop={'size': 12})



        i+=1
    plt.tight_layout()
    fig.savefig(os.path.join(savedir,'{}_ET.png'.format(frequency)),dpi=300,bbox_inches='tight')
    fig.savefig(os.path.join(savedir,'{}_ET.svg'.format(frequency)),dpi=300,bbox_inches='tight')
    fig.savefig(os.path.join(savedir,'{}_ET.pdf'.format(frequency)),dpi=300,bbox_inches='tight')

if frequency == 'monthly':

    # Monthly plot
    gdf_metrics = gpd.read_parquet(os.path.join(metrics_dir,'{}_utm12n_nad83.parquet.gzip'.format(frequency)))
    df_fluxnet = pd.read_parquet(data_dict['ET']['FLUXNET_GAP'])
    df_fluxnet_corr = pd.read_parquet(data_dict['ET']['FLUXNET_CORR'])
    df_nwm = pd.read_parquet(data_dict['ET']['NWM'])


    # Convert the 'igbp' column to a categorical data type
    gdf_metrics['igbp'] = pd.Categorical(gdf_metrics['igbp'], categories=igbp_order, ordered=True)

    # Sort the DataFrame by 'igbp'
    gdf_metrics = gdf_metrics.sort_values('igbp')


    fig, axs = plt.subplots(3,3,figsize=(8,8),sharex=True,sharey=True)
    i=0
    for ax in axs.flatten():
        station = gdf_metrics.iloc[i].name
        igbp = gdf_metrics.loc[station,'igbp']
        df_fluxnet_station = df_fluxnet.loc[:,station]
        df_fluxnet_corr_station = df_fluxnet_corr.loc[:,station]
        df_nwm_station = df_nwm.loc[:,station]

        df_fluxnet_station = df_fluxnet_station.dropna()
        df_fluxnet_station = df_fluxnet_station.resample('ME').sum()

        df_fluxnet_corr_station = df_fluxnet_corr_station.dropna()
        df_fluxnet_corr_station = df_fluxnet_corr_station.resample('ME').sum()

        start_year = df_fluxnet_corr_station.index[0].year
        end_year = df_fluxnet_corr_station.index[-1].year

        df_nwm_station = df_nwm_station.dropna()
        df_nwm_station = df_nwm_station.resample('ME').sum()

        df_nwm_station = df_nwm_station.loc[df_fluxnet_station.index]
        # Convert to US/Arizona Timezone
        df_fluxnet_station.index = df_fluxnet_station.index.tz_localize('UTC').tz_convert('US/Arizona')
        df_fluxnet_corr_station.index = df_fluxnet_corr_station.index.tz_localize('UTC').tz_convert('US/Arizona')
        df_nwm_station.index = df_nwm_station.index.tz_localize('UTC').tz_convert('US/Arizona')


        dummy_dt_index = pd.date_range(start='2020-01-01',end='2020-12-31',freq='ME')
        # Calc Mean
        df_fluxnet_corr_mean = df_fluxnet_corr_station.groupby(df_fluxnet_corr_station.index.month).mean()
        df_nwm_mean = df_nwm_station.groupby(df_nwm_station.index.month).mean()
        df_mean = pd.concat([df_fluxnet_corr_mean,df_nwm_mean],axis=1)
        df_mean.columns = ['FLUXNET','NWM']
        df_mean.index = dummy_dt_index


        # Calc Std
        df_fluxnet_corr_std = df_fluxnet_corr_station.groupby(df_fluxnet_corr_station.index.month).std()
        df_nwm_std = df_nwm_station.groupby(df_nwm_station.index.month).std()
        df_std = pd.concat([df_fluxnet_corr_std,df_nwm_std],axis=1)
        df_std.columns = ['FLUXNET','NWM']
        df_std.index = dummy_dt_index

        # Calc 5-95 Percentile
        df_fluxnet_corr_5 = df_fluxnet_corr_station.groupby(df_fluxnet_corr_station.index.month).quantile(0.05)
        df_nwm_5 = df_nwm_station.groupby(df_nwm_station.index.month).quantile(0.05)
        df_5 = pd.concat([df_fluxnet_corr_5,df_nwm_5],axis=1)
        df_5.columns = ['FLUXNET','NWM']
        df_5.index = dummy_dt_index

        df_fluxnet_corr_95 = df_fluxnet_corr_station.groupby(df_fluxnet_corr_station.index.month).quantile(0.95)
        df_nwm_95 = df_nwm_station.groupby(df_nwm_station.index.month).quantile(0.95)
        df_95 = pd.concat([df_fluxnet_corr_95,df_nwm_95],axis=1)
        df_95.columns = ['FLUXNET','NWM']
        df_95.index = dummy_dt_index

        # Plot
        df_mean.loc[:,'FLUXNET'].plot(ax=ax,color='k',ls='-',lw=0.8,label='FLUXNET')
        # ax.fill_between(df_mean.index,df_mean['FLUXNET']-df_std['FLUXNET'],df_mean['FLUXNET']+df_std['FLUXNET'],color='k',alpha=0.2)
        ax.fill_between(df_mean.index,df_5['FLUXNET'],df_95['FLUXNET'],color='k',alpha=0.2,lw=0.5)

        df_mean.loc[:,'NWM'].plot(ax=ax,color='r',ls='--',lw=0.8,label='NWM')
        # ax.fill_between(df_mean.index,df_mean['NWM']-df_std['NWM'],df_mean['NWM']+df_std['NWM'],color='r',alpha=0.2)
        ax.fill_between(df_mean.index,df_5['NWM'],df_95['NWM'],color='r',alpha=0.2,lw=0.5)

        ax.set_xticks(dummy_dt_index)
        ax.set_xticklabels(dummy_dt_index.strftime('%b'))
        fmt = ax.xaxis.get_major_formatter()
        ax.xaxis.set_major_formatter(lambda x, pos: fmt(x, pos)[0])

        # Remove minor ticks
        ax.xaxis.set_minor_locator(NullLocator())
        ax.yaxis.set_minor_locator(NullLocator())
        


        ax.grid(True,ls='--',color='gray',alpha=0.5)
        ax.set_title(station+' ({})'.format(igbp),fontsize=14)
        ax.tick_params(axis='both', which='major', labelsize=12)
        ax.set_ylabel('$ET$ (mm/day)',fontsize=12)



        NNSE = gdf_metrics.loc[station,'ET_NNSE_C_WY']
        PBIAS = gdf_metrics.loc[station,'ET_PBIAS_C_WY']
        RMSE = gdf_metrics.loc[station,'ET_RMSE_C_WY']
        CC = gdf_metrics.loc[station,'ET_PEARSON_C_WY']

        # Show NNSE, PBIAS, RMSE, and CC in the top left corner of the subplot
        # ax.text(0.02, 0.98, f'RMSE: {RMSE:.1f}\nNNSE: {NNSE:.2f}\nPBIAS: {PBIAS:.1f}\n$r$: {CC:.2f}', 
        #         transform=ax.transAxes, verticalalignment='top', fontsize=12)
        ax.text(0.02, 0.98, f'RMSE: {RMSE:.1f}\nNNSE: {NNSE:.2f}\nPBIAS: {PBIAS:.1f}', 
                transform=ax.transAxes, verticalalignment='top', fontsize=12)
        ax.text(0.01, 1.11, f'({chr(97 + i)})', transform=ax.transAxes, verticalalignment='top', fontsize=14)
        ax.text(0.98, 0.98, f'{start_year}-{end_year}', 
                transform=ax.transAxes, horizontalalignment='right', verticalalignment='top', fontsize=12)



        # Create a legend with a dashed line inside a shaded region
        fluxnet_line = mlines.Line2D([], [], color='k', linestyle='-', linewidth=0.8, label='FLUXNET')
        fluxnet_patch = mpatches.Patch(color='k', alpha=0.2, label='FLUXNET')
        nwm_line = mlines.Line2D([], [], color='r', linestyle='--', linewidth=0.8, label='NWM')
        nwm_patch = mpatches.Patch(color='r', alpha=0.2, label='NWM')

        if i==0:
            # Add a legend for the whole figure below the bottom of the figure
            fig.legend(handles=[fluxnet_line, fluxnet_patch, nwm_line, nwm_patch], loc='lower center', ncol=2, bbox_to_anchor=(0.5, -0.07),prop={'size': 12})




        i+=1
    plt.tight_layout()
    fig.savefig(os.path.join(savedir,'{}_ET.png'.format(frequency)),dpi=300,bbox_inches='tight')
    fig.savefig(os.path.join(savedir,'{}_ET.svg'.format(frequency)),dpi=300,bbox_inches='tight')
    fig.savefig(os.path.join(savedir,'{}_ET.pdf'.format(frequency)),dpi=300,bbox_inches='tight')
