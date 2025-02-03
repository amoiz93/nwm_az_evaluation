import os
import sys
sys.path.append('../src')
import glob
import ast

import xarray as xr
import numpy as np
import pandas as pd
import geopandas as gpd

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Patch
import matplotlib

import eval_metrics
import misc
import param_nwm3
import usgs

# matplotlib.use('cairo')

savedir = os.path.join(param_nwm3.out_dir,'figures',os.path.basename(__file__).split('.')[0])
misc.makedir(savedir)

gdf_metrics = gpd.read_parquet('/Users/amoiz2/Servers/dropbox_asu/Personal/Work/20230807_ASU_SSEBE_Mascaro/Workspace/projects/20240311_USGS_NWM_Q_Validation_AZ/out/06A_add_attributes/parquet/daily_2003_2022_utm12n_nad83.parquet.gzip')
usgs_data_dir = '../out/03_reformat_USGS_Q_data/usgs/parquet/'
nwm_data_dir = '../out/04_nwm3_data_download/nwm3/parquet/'

frequency = 'daily'

# Cases
site_ids = [
        #    '09431500',#'09499000',#'09498501',#'09505800',
            '09444200',#'09499000',#'09432000',
        #    '09471550',#(1)'09484000'#'09471550',#'09471000',#'09486055',#'09492400'
           '09444000',
           '09489500'
           ]

seasons = {
           'all':{'months':[1,2,3,4,5,6,7,8,9,10,11,12],'symbol':'A'},   # Jan - Dec
           'winter':{'months':[11,12,1,2,3],'symbol':'W'}, # Nov - Mar
           'summer':{'months':[7,8,9],'symbol':'S'} # Jul - Sep
          }

start_date = '2015-10-01'
end_date = '2017-09-30'

start_date_fdc = '2002-10-01'
end_date_fdc = '2022-09-30'


axis_label_fontsize = 12

# Create patches for the legend
blue_patch = Patch(color='lightblue', alpha=0.5, label='W')
coral_patch = Patch(color='lightcoral', alpha=0.5, label='S')
lw=0.6

# Station 1
fig,ax = plt.subplots(3,3,figsize=(10,7))

i=0
for site_id in site_ids:
    comid = int(gdf_metrics.loc[gdf_metrics['USGS_ID']==site_id,'nwm3g_comid'])
    sel_years_DA = ast.literal_eval(gdf_metrics.loc[gdf_metrics['USGS_ID']==site_id,'sel_years_DA'].values[0])
    # Read Data
    df_usgs = pd.read_parquet(os.path.join(usgs_data_dir,frequency,'{}.parquet.gzip'.format(str(site_id))))
    df_nwm = pd.read_parquet(os.path.join(nwm_data_dir,frequency,'{}.parquet.gzip'.format(str(comid))))

    df_usgs_orig = usgs.add_water_year(df_usgs.copy())
    df_nwm_orig = usgs.add_water_year(df_nwm.copy())

    df_usgs_orig = df_usgs_orig[df_usgs_orig['WY'].isin(sel_years_DA)]
    df_nwm_orig = df_nwm_orig[df_nwm_orig['WY'].isin(sel_years_DA)]

    df_usgs_orig = df_usgs_orig.drop(columns=['WY']).iloc[:,0]
    df_nwm_orig = df_nwm_orig.drop(columns=['WY']).iloc[:,0]

    df_usgs_orig = df_usgs_orig[df_usgs_orig.notnull()&(df_nwm_orig.notnull())]
    df_nwm_orig = df_nwm_orig[df_usgs_orig.notnull()&(df_nwm_orig.notnull())]

    df_usgs_orig = pd.DataFrame(df_usgs_orig)
    df_nwm_orig = pd.DataFrame(df_nwm_orig)

    # df_usgs_orig = df_usgs_orig[(df_usgs_orig['water_year']>=sel_years_DA[0]) & (df_usgs_orig['water_year']<=sel_years_DA[1])]
    # df_usgs_orig = df_usgs.copy()[(df_usgs.index>=start_date_fdc) & (df_usgs.index<=end_date_fdc)]
    # df_nwm_orig = df_nwm.copy()[(df_nwm.index>=start_date_fdc) & (df_nwm.index<=end_date_fdc)]

    df_usgs = df_usgs[(df_usgs.index>=start_date) & (df_usgs.index<=end_date)]
    df_nwm = df_nwm[(df_nwm.index>=start_date) & (df_nwm.index<=end_date)]

    df_usgs.columns = ['USGS']
    df_nwm.columns = ['NWM']

    # Plot 1
    df_usgs.plot(ax=ax[0,i], color='k',label='USGS',lw=lw)
    df_nwm.plot(ax=ax[0,i], color='r',label='NWM',ls='--',lw=lw)
    ax[0,i].set_ylim(bottom=0)
    if i==0:
        ax[0,i].set_ylabel('Q (m$^3$/s)',fontsize=axis_label_fontsize)
    if i==2:
        # ax[0,i].set_ylabel('Q (m$^3$/s)',fontsize=axis_label_fontsize)
        handles, labels = ax[0,i].get_legend_handles_labels()
        handles.extend([blue_patch, coral_patch])
        ax[0,i].legend(handles=handles,ncols=1,framealpha=1.0,prop={'size': 8})
    else:
        ax[0,i].get_legend().remove()

    NNSE_str = str(gdf_metrics.loc[gdf_metrics['USGS_ID']==site_id,'NNSE_A'].values[0])
    LNNSE_str = str(gdf_metrics.loc[gdf_metrics['USGS_ID']==site_id,'lnNNSE_A'].values[0])
    NNSE_S_str = str(gdf_metrics.loc[gdf_metrics['USGS_ID']==site_id,'NNSE_S'].values[0])
    LNNSE_S_str = str(gdf_metrics.loc[gdf_metrics['USGS_ID']==site_id,'lnNNSE_S'].values[0])
    NNSE_W_str = str(gdf_metrics.loc[gdf_metrics['USGS_ID']==site_id,'NNSE_W'].values[0])
    LNNSE_W_str = str(gdf_metrics.loc[gdf_metrics['USGS_ID']==site_id,'lnNNSE_W'].values[0])
    Basin = gdf_metrics.loc[gdf_metrics['USGS_ID']==site_id,'M_BASIN_ABR'].values[0]
    Area = round(gdf_metrics.loc[gdf_metrics['USGS_ID']==site_id,'DAREA_km2'].values[0],1)
    
    if i==0:
        condition_str = 'NNSE in S < NNSE in W'
        metric_str = 'NNSE in WY: {} ({})\nNNSE in S: {} ({})\nNNSE in W: {} ({})'.format(NNSE_str,LNNSE_str,NNSE_S_str,LNNSE_S_str,NNSE_W_str,LNNSE_W_str)
    elif i==1:
        condition_str = 'NNSE in S > NNSE in W'
        metric_str = 'NNSE in WY: {} ({})\nNNSE in S: {} ({})\nNNSE in W: {} ({})'.format(NNSE_str,LNNSE_str,NNSE_S_str,LNNSE_S_str,NNSE_W_str,LNNSE_W_str)
    else:
        condition_str = 'NNSE in WY < LNNSE in WY'
        metric_str = 'NNSE in WY: {} ({})\nNNSE in S: {} ({})\nNNSE in W: {} ({})'.format(NNSE_str,LNNSE_str,NNSE_S_str,LNNSE_S_str,NNSE_W_str,LNNSE_W_str)
    # ax[0,i].set_title('USGS ID: {} \n(Basin: {}; Area: {} km$^2$)\nNNSE,LNNSE[WY]:{},{}\nNNSE,LNNSE[S]:{},{}\nNNSE,LNNSE[W]:{},{}'.format(site_id,Basin,Area,NNSE_str,LNNSE_str,LNNSE_S_str,NNSE_S_str,NNSE_W_str,LNNSE_W_str))
    #ax[0,i].set_title('USGS ID: {} \n(Basin: {}; Area: {} km$^2$)'.format(site_id,Basin,Area))
    ax[0,i].set_title('USGS ID: {} (Basin: {})'.format(site_id,Basin))
    # ax[2,i].text(0.02,0.02,'NNSE,LNNSE[WY]:{},{}\nNNSE,LNNSE[S]:{},{}\nNNSE,LNNSE[W]:{},{}'.format(NNSE_str,LNNSE_str,LNNSE_S_str,NNSE_S_str,NNSE_W_str,LNNSE_W_str),transform=ax[2,i].transAxes,fontsize=8)

    if i == 0:
        ax[2,i].text(0.40,0.97,metric_str,transform=ax[2,i].transAxes,fontsize=9,ha='left',va='top')
    else:
        ax[2,i].text(0.01,0.02,metric_str,transform=ax[2,i].transAxes,fontsize=9)
    ax[0,i].text(0.5,1.2,'({})'.format(chr(97+i)),transform=ax[0,i].transAxes,fontsize=14,ha='center',weight='bold')
    # ax[0,i].text(0.02,0.85,'({})'.format(chr(97+i)),transform=ax[0,i].transAxes,fontsize=12)
    
    wy_list = list(df_nwm.index.year.unique())[1:]
    for str_wy in wy_list:
        start_shade = mdates.date2num(pd.to_datetime('{}-11-01'.format(str(str_wy-1))))  # replace YYYY with the year
        end_shade = mdates.date2num(pd.to_datetime('{}-03-31'.format(str(str_wy))))  # replace YYYY with the year
        ax[0,i].fill_betweenx(ax[0,i].get_ylim(), start_shade, end_shade, facecolor='lightblue', alpha=0.5)
        
        start_shade = mdates.date2num(pd.to_datetime('{}-07-01'.format(str(str_wy))))  # replace YYYY with the year
        end_shade = mdates.date2num(pd.to_datetime('{}-09-30'.format(str(str_wy))))  # replace YYYY with the year
        ax[0,i].fill_betweenx(ax[0,i].get_ylim(), start_shade, end_shade, facecolor='lightcoral', alpha=0.5)
    

    # Plot 2
    df_usgs.plot(ax=ax[1,i], color='k',label='USGS',lw=lw)
    df_nwm.plot(ax=ax[1,i], color='r',label='NWM',ls='--',lw=lw)
    max_val = max(df_usgs.max().values[0],df_nwm.max().values[0])
    wy_list = list(df_nwm.index.year.unique())[1:]
    for str_wy in wy_list:
        start_shade = mdates.date2num(pd.to_datetime('{}-11-01'.format(str(str_wy-1))))  # replace YYYY with the year
        end_shade = mdates.date2num(pd.to_datetime('{}-03-31'.format(str(str_wy))))  # replace YYYY with the year
        ax[1,i].fill_betweenx((ax[1,i].get_ylim()[0],max_val*1.2), start_shade, end_shade, facecolor='lightblue', alpha=0.5)
        
        start_shade = mdates.date2num(pd.to_datetime('{}-07-01'.format(str(str_wy))))  # replace YYYY with the year
        end_shade = mdates.date2num(pd.to_datetime('{}-09-30'.format(str(str_wy))))  # replace YYYY with the year
        ax[1,i].fill_betweenx((ax[1,i].get_ylim()[0],max_val*1.2), start_shade, end_shade, facecolor='lightcoral', alpha=0.5)
    
    
    ax[1,i].set_yscale('log')
    ax[1,i].set_ylim((ax[1,i].get_ylim()[0],max_val*1.2))
    if i==0:
        ax[1,i].set_ylabel('Q (m$^3$/s) - log scale',fontsize=axis_label_fontsize)
    # ax[1,i].text(0.02,0.85,'({})'.format(chr(100+i)),transform=ax[1,i].transAxes,fontsize=12)
    ax[1,i].get_legend().remove()
    


    # Plot 3
    df_usgs_fdc = eval_metrics.FDC(df_usgs_orig)
    df_nwm_fdc = eval_metrics.FDC(df_nwm_orig)

    df_usgs_fdc_S = eval_metrics.FDC(df_usgs_orig[df_usgs_orig.index.month.isin(seasons['summer']['months'])])
    df_nwm_fdc_S = eval_metrics.FDC(df_nwm_orig[df_nwm_orig.index.month.isin(seasons['summer']['months'])])

    df_usgs_fdc_W = eval_metrics.FDC(df_usgs_orig[df_usgs_orig.index.month.isin(seasons['winter']['months'])])
    df_nwm_fdc_W = eval_metrics.FDC(df_nwm_orig[df_nwm_orig.index.month.isin(seasons['winter']['months'])])

    max_val = max(df_usgs_fdc.max().values[0],df_nwm_fdc.max().values[0])

    # WY
    df_usgs_fdc.plot(ax=ax[2,i],x='exceedance',y='Q', logx=False,logy=True, color='k',label='USGS in WY',lw=lw)
    df_nwm_fdc.plot(ax=ax[2,i],x='exceedance',y='Q', logx=False,logy=True, color='k',label='NWM in WY',lw=lw,ls='--')
    
    # Summer
    df_usgs_fdc_S.plot(ax=ax[2,i],x='exceedance',y='Q', logx=False,logy=True, color='r',label='USGS in S',lw=lw)
    df_nwm_fdc_S.plot(ax=ax[2,i],x='exceedance',y='Q', logx=False,logy=True, color='r',label='NWM in S',lw=lw,ls='--')

    # Winter
    df_usgs_fdc_W.plot(ax=ax[2,i],x='exceedance',y='Q', logx=False,logy=True, color='b',label='USGS in W',lw=lw)
    df_nwm_fdc_W.plot(ax=ax[2,i],x='exceedance',y='Q', logx=False,logy=True, color='b',label='NWM in W',lw=lw,ls='--')

    if i==0:
        ax[2,i].set_ylabel('Q (m$^3$/s) - log scale',fontsize=axis_label_fontsize)
    ax[2,i].set_xlabel('Flow exceedance (%)',fontsize=axis_label_fontsize)
    # ax[2,i].text(0.90,0.85,'({})'.format(chr(103+i)),transform=ax[2,i].transAxes,fontsize=12)
    
    if i ==2:
        ax[2,i].legend(loc='upper right',ncol=2,framealpha=1.0,prop={'size': 8},columnspacing=0.5)
    else:
        ax[2,i].get_legend().remove()
    # ax[2,i].fill_betweenx((ax[2,i].get_ylim()[0],max_val*1.2), 0, 2, facecolor='lightcoral', alpha=0.5)
    # ax[2,i].fill_betweenx((ax[2,i].get_ylim()[0],max_val*1.2), 70, 100, facecolor='lightblue', alpha=0.5)
    i+=1





plt.tight_layout()
fig.savefig(os.path.join(savedir,'hydrograph_samples.png'),dpi=300)
fig.savefig(os.path.join(savedir,'hydrograph_samples.svg'),dpi=300)
fig.savefig(os.path.join(savedir,'hydrograph_samples.pdf'),dpi=300)


