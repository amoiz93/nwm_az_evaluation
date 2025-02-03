import os
import sys
sys.path.append('../src')
import glob
import ast

import xarray as xr
import numpy as np
import pandas as pd
import geopandas as gpd
import pyogrio

import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib as mpl

import misc
import param_nwm3

import custom_basemaps as cbm

def non_missing_WYs_between(df,col_name,start_wy,end_wy):
    non_missing = df[col_name]
    non_missing_df = []
    for i in range(0,len(non_missing)):
        non_missing_station = pd.DataFrame(ast.literal_eval(non_missing.iloc[i])).T
        non_missing_station.index = [df['USGS_ID'].iloc[i]]
        non_missing_df.append(non_missing_station)
    non_missing_df = pd.concat(non_missing_df)
    total_non_missing_2003_2022 = non_missing_df[(non_missing_df>=start_wy)&(non_missing_df<=end_wy)].count(axis=1).sum()
    total_non_missing = non_missing_df.count(axis=1).sum()
    total_non_missing_percentage = (total_non_missing_2003_2022/total_non_missing)*100
    return total_non_missing_percentage

def calc_nnse_threshold(gdf_metrics,threshold):
    gdf_metrics = gdf_metrics[['NNSE_A','NNSE_S','NNSE_W','lnNNSE_A','lnNNSE_S','lnNNSE_W']]
    gdf_metrics = (gdf_metrics[gdf_metrics>threshold].count()/gdf_metrics.count())*100
    return gdf_metrics

def calc_nnse_threshold_clim(gdf_metrics,threshold):
    df = gdf_metrics[['NNSE_A','NNSE_S','NNSE_W','lnNNSE_A','lnNNSE_S','lnNNSE_W']]
    gdf_metrics = (df[df>threshold].groupby(gdf_metrics['Climate']).count()/df.groupby(gdf_metrics['Climate']).count()) * 100
    return gdf_metrics

def calc_pbias_threshold(gdf_metrics,threshold):
    gdf_metrics = gdf_metrics[['PBIAS_A','PBIAS_S','PBIAS_W','lnPBIAS_A','lnPBIAS_S','lnPBIAS_W']]
    gdf_metrics = (gdf_metrics[gdf_metrics.abs()<=threshold].count()/gdf_metrics.count())*100
    return gdf_metrics

def calc_pbias_threshold_positive(gdf_metrics,threshold):
    gdf_metrics = gdf_metrics[['PBIAS_A','PBIAS_S','PBIAS_W','lnPBIAS_A','lnPBIAS_S','lnPBIAS_W']]
    gdf_metrics = (gdf_metrics[(gdf_metrics.abs()<=threshold) & (gdf_metrics>0)].count()/gdf_metrics.count())*100
    return gdf_metrics

def calc_pbias_threshold_negative(gdf_metrics,threshold):
    gdf_metrics = gdf_metrics[['PBIAS_A','PBIAS_S','PBIAS_W','lnPBIAS_A','lnPBIAS_S','lnPBIAS_W']]
    gdf_metrics = (gdf_metrics[(gdf_metrics.abs()<=threshold) & (gdf_metrics<0)].count()/gdf_metrics.count())*100
    return gdf_metrics

def calc_pbias_threshold_clim(gdf_metrics,threshold):
    df = gdf_metrics[['PBIAS_A','PBIAS_S','PBIAS_W','lnPBIAS_A','lnPBIAS_S','lnPBIAS_W']]
    gdf_metrics = (df[df.abs()<=threshold].groupby(gdf_metrics['Climate']).count()/df.groupby(gdf_metrics['Climate']).count()) * 100
    return gdf_metrics

def calc_pbias_threshold_clim_positive(gdf_metrics,threshold):
    df = gdf_metrics[['PBIAS_A','PBIAS_S','PBIAS_W','lnPBIAS_A','lnPBIAS_S','lnPBIAS_W']]
    gdf_metrics = (df[(df.abs()<=threshold) & (df>0)].groupby(gdf_metrics['Climate']).count()/df.groupby(gdf_metrics['Climate']).count()) * 100
    return gdf_metrics

def calc_pbias_threshold_clim_negative(gdf_metrics,threshold):
    df = gdf_metrics[['PBIAS_A','PBIAS_S','PBIAS_W','lnPBIAS_A','lnPBIAS_S','lnPBIAS_W']]
    gdf_metrics = (df[(df.abs()<=threshold) & (df<0)].groupby(gdf_metrics['Climate']).count()/df.groupby(gdf_metrics['Climate']).count()) * 100
    return gdf_metrics




def metrics_threshold(gdf_metrics,NNSE_threshold,PBIAS_threshold):
    nnse_threshold_all = calc_nnse_threshold(gdf_metrics,NNSE_threshold)
    pbias_threshold_all = calc_pbias_threshold(gdf_metrics,PBIAS_threshold)

    pbias_positive_threshold_all = calc_pbias_threshold_positive(gdf_metrics,PBIAS_threshold)
    pbias_negative_threshold_all = calc_pbias_threshold_negative(gdf_metrics,PBIAS_threshold)

    nnse_threshold_all.name = 'ALL'
    pbias_threshold_all.name = 'ALL'
    pbias_positive_threshold_all.name = 'ALL'
    pbias_negative_threshold_all.name = 'ALL'

    nnse_threshold_all = pd.DataFrame(nnse_threshold_all).T
    pbias_threshold_all = pd.DataFrame(pbias_threshold_all).T
    pbias_positive_threshold_all = pd.DataFrame(pbias_positive_threshold_all).T
    pbias_negative_threshold_all = pd.DataFrame(pbias_negative_threshold_all).T

    pbias_positive_threshold_all = pbias_positive_threshold_all.drop(columns=['lnPBIAS_A','lnPBIAS_S','lnPBIAS_W']) 
    pbias_negative_threshold_all = pbias_negative_threshold_all.drop(columns=['lnPBIAS_A','lnPBIAS_S','lnPBIAS_W']) 

    pbias_positive_threshold_all = pbias_positive_threshold_all.rename(columns={'PBIAS_A':'PBIAS_P_WY','PBIAS_S':'PBIAS_P_S','PBIAS_W':'PBIAS_P_W'})
    pbias_negative_threshold_all = pbias_negative_threshold_all.rename(columns={'PBIAS_A':'PBIAS_N_WY','PBIAS_S':'PBIAS_N_S','PBIAS_W':'PBIAS_N_W'})

    nnse_threshold_clim_group = calc_nnse_threshold_clim(gdf_metrics,NNSE_threshold)
    pbias_threshold_clim_group = calc_pbias_threshold_clim(gdf_metrics,PBIAS_threshold)
    pbias_positive_threshold_clim_group = calc_pbias_threshold_clim_positive(gdf_metrics,PBIAS_threshold)
    pbias_negative_threshold_clim_group = calc_pbias_threshold_clim_negative(gdf_metrics,PBIAS_threshold)

    nnse_threshold_clim_group.index = nnse_threshold_clim_group.index.map({'M':'MON','O':'OTH','W':'WIN'})
    pbias_threshold_clim_group.index = pbias_threshold_clim_group.index.map({'M':'MON','O':'OTH','W':'WIN'})
    pbias_positive_threshold_clim_group.index = pbias_positive_threshold_clim_group.index.map({'M':'MON','O':'OTH','W':'WIN'})
    pbias_negative_threshold_clim_group.index = pbias_negative_threshold_clim_group.index.map({'M':'MON','O':'OTH','W':'WIN'})
    pbias_positive_threshold_clim_group = pbias_positive_threshold_clim_group.drop(columns=['lnPBIAS_A','lnPBIAS_S','lnPBIAS_W'])
    pbias_negative_threshold_clim_group = pbias_negative_threshold_clim_group.drop(columns=['lnPBIAS_A','lnPBIAS_S','lnPBIAS_W'])
    pbias_positive_threshold_clim_group = pbias_positive_threshold_clim_group.rename(columns={'PBIAS_A':'PBIAS_P_WY','PBIAS_S':'PBIAS_P_S','PBIAS_W':'PBIAS_P_W'})
    pbias_negative_threshold_clim_group = pbias_negative_threshold_clim_group.rename(columns={'PBIAS_A':'PBIAS_N_WY','PBIAS_S':'PBIAS_N_S','PBIAS_W':'PBIAS_N_W'})

    nnse_threshold = pd.concat([nnse_threshold_all,nnse_threshold_clim_group])
    pbias_threshold = pd.concat([pbias_threshold_all,pbias_threshold_clim_group])
    pbias_positive_threshold = pd.concat([pbias_positive_threshold_all,pbias_positive_threshold_clim_group])
    pbias_negative_threshold = pd.concat([pbias_negative_threshold_all,pbias_negative_threshold_clim_group])

    metrics_threshold = pd.concat([nnse_threshold,pbias_threshold,pbias_positive_threshold,pbias_negative_threshold],axis=1)
    metrics_threshold.columns = metrics_threshold.columns.map({'NNSE_A':'NNSE_WY','NNSE_S':'NNSE_S','NNSE_W':'NNSE_W','lnNNSE_A':'LNNSE_WY','lnNNSE_S':'LNNSE_S','lnNNSE_W':'LNNSE_W','PBIAS_A':'PBIAS_WY','PBIAS_S':'PBIAS_S','PBIAS_W':'PBIAS_W','lnPBIAS_A':'LPBIAS_WY','lnPBIAS_S':'LPBIAS_S','lnPBIAS_W':'LPBIAS_W','PBIAS_P_WY':'PBIAS_P_WY','PBIAS_P_S':'PBIAS_P_S','PBIAS_P_W':'PBIAS_P_W','PBIAS_N_WY':'PBIAS_N_WY','PBIAS_N_S':'PBIAS_N_S','PBIAS_N_W':'PBIAS_N_W'})
    metrics_threshold = metrics_threshold.round(1)
    metrics_threshold = metrics_threshold.reindex(['ALL','MON','WIN','OTH'])
    metrics_threshold = metrics_threshold.drop(columns=['LPBIAS_WY','LPBIAS_S','LPBIAS_W'])
    return metrics_threshold




def calc_nnse_threshold_basin(gdf_metrics,threshold):
    df = gdf_metrics[['NNSE_A','NNSE_S','NNSE_W','lnNNSE_A','lnNNSE_S','lnNNSE_W']]
    gdf_metrics = (df[df>threshold].groupby(gdf_metrics['M_BASIN_ABR']).count()/df.groupby(gdf_metrics['M_BASIN_ABR']).count()) * 100
    return gdf_metrics

def calc_pbias_threshold_basin(gdf_metrics,threshold):
    df = gdf_metrics[['PBIAS_A','PBIAS_S','PBIAS_W','lnPBIAS_A','lnPBIAS_S','lnPBIAS_W']]
    gdf_metrics = (df[df.abs()<=threshold].groupby(gdf_metrics['M_BASIN_ABR']).count()/df.groupby(gdf_metrics['M_BASIN_ABR']).count()) * 100
    return gdf_metrics

def sorted_metric(gdf_metrics,NNSE_threshold,PBIAS_threshold):
    metrics = ['NNSE_A','NNSE_S','NNSE_W','lnNNSE_A','lnNNSE_S','lnNNSE_W','PBIAS_A','PBIAS_S','PBIAS_W']
    sorted_metric = {}
    for metric in metrics:
        if metrics.index(metric) < 6:
            sorted_metric[metric] = calc_nnse_threshold_basin(gdf_metrics,NNSE_threshold).sort_values(metric,ascending=False)[metric]
        else:
            sorted_metric[metric] = calc_pbias_threshold_basin(gdf_metrics,PBIAS_threshold).sort_values(metric,ascending=False)[metric]
    return sorted_metric

def count_values_PBIAS_HF_LF(df,threshold_hf,threshold_lf):
    PBIAS_HF_A_count = df[(df['PBIAS_HF_A'].abs()>threshold_hf) | np.isinf(df['PBIAS_HF_A'])].groupby('Climate').count()['PBIAS_HF_A']/df.count()['PBIAS_HF_A']*100
    PBIAS_HF_S_count = df[(df['PBIAS_HF_S'].abs()>threshold_hf) | np.isinf(df['PBIAS_HF_S'])].groupby('Climate').count()['PBIAS_HF_S']/df.count()['PBIAS_HF_S']*100
    PBIAS_HF_W_count = df[(df['PBIAS_HF_W'].abs()>threshold_hf) | np.isinf(df['PBIAS_HF_W'])].groupby('Climate').count()['PBIAS_HF_W']/df.count()['PBIAS_HF_W']*100

    PBIAS_LF_A_count = df[(df['PBIAS_LF_A'].abs()>threshold_lf) | np.isinf(df['PBIAS_LF_A'])].groupby('Climate').count()['PBIAS_LF_A']/df.count()['PBIAS_LF_A']*100
    PBIAS_LF_S_count = df[(df['PBIAS_LF_S'].abs()>threshold_lf) | np.isinf(df['PBIAS_LF_S'])].groupby('Climate').count()['PBIAS_LF_S']/df.count()['PBIAS_LF_S']*100
    PBIAS_LF_W_count = df[(df['PBIAS_LF_W'].abs()>threshold_lf) | np.isinf(df['PBIAS_LF_W'])].groupby('Climate').count()['PBIAS_LF_W']/df.count()['PBIAS_LF_W']*100

    df_out = pd.concat([PBIAS_HF_A_count,
                        PBIAS_HF_S_count,
                        PBIAS_HF_W_count,
                        PBIAS_LF_A_count,
                        PBIAS_LF_S_count,
                        PBIAS_LF_W_count],axis=1)
    df_out = df_out.rename(columns={'PBIAS_HF_A':'PBIAS$_\mathrm{HF}$ [WY]',
                                    'PBIAS_HF_S':'PBIAS$_\mathrm{HF}$ [S]',
                                    'PBIAS_HF_W':'PBIAS$_\mathrm{HF}$ [W]',
                                    'PBIAS_LF_A':'PBIAS$_\mathrm{LF}$ [WY]',
                                    'PBIAS_LF_S':'PBIAS$_\mathrm{LF}$ [S]',
                                    'PBIAS_LF_W':'PBIAS$_\mathrm{LF}$ [W]'})
    df_out = df_out.rename(index={'M':'MON','O':'OTH','W':'WIN'})
    df_out = df_out.reindex(['MON','WIN','OTH'])
    df_out.index.name = 'Basin group'
    return df_out

savedir = os.path.join(param_nwm3.out_dir,'figures',os.path.basename(__file__).split('.')[0])
misc.makedir(savedir)


metrics_dir = '/Users/amoiz2/Servers/dropbox_asu/Personal/Work/20230807_ASU_SSEBE_Mascaro/Workspace/projects/20240311_USGS_NWM_Q_Validation_AZ/out/06A_add_attributes/parquet'
start_year = 2003
end_year = 2022
gdf_site_info = gpd.read_parquet('/Users/amoiz2/Servers/dropbox_asu/Personal/Work/20230807_ASU_SSEBE_Mascaro/Workspace/projects/20240311_USGS_NWM_Q_Validation_AZ/out/07_final_usgs_sites/usgs_q_info_flagged_utm12n_nad83.parquet.gzip')
gdf_metrics_hourly = gpd.read_parquet(os.path.join(metrics_dir,'{}_{}_{}_utm12n_nad83.parquet.gzip'.format('hourly',str(start_year),str(end_year))))
gdf_metrics_daily = gpd.read_parquet(os.path.join(metrics_dir,'{}_{}_{}_utm12n_nad83.parquet.gzip'.format('daily',str(start_year),str(end_year))))
gdf_az_basins_utm12n = pyogrio.read_dataframe(os.path.join('../../../data/az_nwm/const/basins/az_watersheds_simplified_utm12n_nad83.shp')).to_crs(gdf_site_info.crs)
gdf_az_basins_utm12n['geometry'] = gdf_az_basins_utm12n.simplify(1E03)

NNSE_threshold = 0.5
PBIAS_threshold = 50

# Sorting Basins based on performance
sorted_basins_daily = sorted_metric(gdf_metrics_daily,NNSE_threshold,PBIAS_threshold)
sorted_basins_hourly = sorted_metric(gdf_metrics_hourly,NNSE_threshold,PBIAS_threshold)

# Daily
dfs = []
for col in sorted_basins_daily.keys():
    dfs.append(pd.DataFrame(sorted_basins_daily[col]).rename(columns={col:col}))
dfs = pd.concat(dfs,axis=1)
dfs.index.name = 'NAME_ABR'
dfs_sorted_basins_daily = dfs.reset_index()

# Hourly
dfs = []
for col in sorted_basins_hourly.keys():
    dfs.append(pd.DataFrame(sorted_basins_hourly[col]).rename(columns={col:col}))
dfs = pd.concat(dfs,axis=1)
dfs.index.name = 'NAME_ABR'
dfs_sorted_basins_hourly = dfs.reset_index()

# Merge with basin info
gdf_metric_threshold_basins_daily = gdf_az_basins_utm12n.merge(dfs_sorted_basins_daily,on='NAME_ABR',how='left')
gdf_metric_threshold_basins_hourly = gdf_az_basins_utm12n.merge(dfs_sorted_basins_hourly,on='NAME_ABR',how='left')

# Rename columns
gdf_metric_threshold_basins_daily = gdf_metric_threshold_basins_daily.rename(columns={'NNSE_A':'NNSE_WY',
                                                                                      'lnNNSE_A':'LNNSE_WY',
                                                                                      'lnNNSE_S':'LNNSE_S',
                                                                                      'lnNNSE_W':'LNNSE_W',
                                                                                      'PBIAS_A':'PBIAS_WY',})

gdf_metric_threshold_basins_hourly = gdf_metric_threshold_basins_hourly.rename(columns={'NNSE_A':'NNSE_WY',
                                                                                        'lnNNSE_A':'LNNSE_WY',
                                                                                        'lnNNSE_S':'LNNSE_S',
                                                                                        'lnNNSE_W':'LNNSE_W',
                                                                                        'PBIAS_A':'PBIAS_WY',})



plot_prop_daily = {
            'NNSE':
             {'bounds':np.array([0,10,20,30,40,50,60,70,80,90,100]),
              'norm':colors.BoundaryNorm(boundaries=np.array([0,10,20,30,40,50,60,70,80,90,100]), ncolors=256),
              'cmap':'turbo_r'},
            'LNNSE':
                {'bounds':np.array([0,10,20,30,40,50,60,70,80,90,100]),
                'norm':colors.BoundaryNorm(boundaries=np.array([0,10,20,30,40,50,60,70,80,90,100]), ncolors=256),
                'cmap':'turbo_r'},
            'PBIAS':
                {'bounds':np.array([0,10,20,30,40,50,60,70,80,90,100]),
                'norm':colors.BoundaryNorm(boundaries=np.array([0,10,20,30,40,50,60,70,80,90,100]), ncolors=256),
                'cmap':'turbo_r'},
                }

plot_prop_hourly = {
            'NNSE':
             {'bounds':np.array([0,10,20,30,40,50,60,70,80,90,100]),
              'norm':colors.BoundaryNorm(boundaries=np.array([0,10,20,30,40,50,60,70,80,90,100]), ncolors=256),
              'cmap':'turbo_r'},
            'LNNSE':
                {'bounds':np.array([0,10,20,30,40,50,60,70,80,90,100]),
                'norm':colors.BoundaryNorm(boundaries=np.array([0,10,20,30,40,50,60,70,80,90,100]), ncolors=256),
                'cmap':'turbo_r'},
            'PBIAS':
                {'bounds':np.array([0,10,20,30,40,50,60,70,80,90,100]),
                'norm':colors.BoundaryNorm(boundaries=np.array([0,10,20,30,40,50,60,70,80,90,100]), ncolors=256),
                'cmap':'turbo_r'},
                }
subplot_labels = {0:'(a)',1:'(b)',2:'(c)',3:'(d)',4:'(e)',5:'(f)',6:'(g)',7:'(h)',8:'(i)'}

# Plotting Basemap
# ax = cbm.az_huc8_basemap()
nx = 2
ny = 3
axis_label_fontsize = 22
subplot_label_fontsize = 18
frequency = 'hourly'

if frequency == 'daily':
    gdf_metric_threshold_basins = gdf_metric_threshold_basins_daily
    plot_prop = plot_prop_daily
if frequency == 'hourly':
    gdf_metric_threshold_basins = gdf_metric_threshold_basins_hourly
    plot_prop = plot_prop_hourly

gdf_climate_boundary = gdf_metric_threshold_basins
gdf_climate_boundary['geometry'] = gdf_climate_boundary['geometry'].buffer(500)
gdf_climate_boundary = gdf_climate_boundary.dissolve('Climate')
gdf_climate_boundary['geometry'] = gdf_climate_boundary['geometry'].buffer(-500)

# NNSE
fig,axs = plt.subplots(nx,ny,subplot_kw={'projection':param_nwm3.cartopy_crs_atur_utm12n},figsize=(10,10))


i=0
for ax in axs.reshape(-1): 
    #ax=cbm.az_huc8_basemap(ax=ax,gridlines_flag=False)

    if i==0:
        metric_name = 'NNSE'
        season_name = 'S'
    if i==1:
        metric_name = 'LNNSE'
        season_name = 'S'
    if i==2:
        metric_name = 'PBIAS'
        season_name = 'S'
    if i==3:
        metric_name = 'NNSE'
        season_name = 'W'
    if i==4:
        metric_name = 'LNNSE'
        season_name = 'W'
    if i==5:
        metric_name = 'PBIAS'
        season_name = 'W'
    
    gdf_metric_threshold_basins.plot(column=f'{metric_name}_{season_name}',ax=ax,cmap=plot_prop[f'{metric_name}']['cmap'],norm=plot_prop[f'{metric_name}']['norm'],legend=False,zorder=100,edgecolor='None')
    gdf_metric_threshold_basins[gdf_metric_threshold_basins['NNSE_WY'].isnull()].plot(facecolor='none',linewidth=0.2,ax=ax,zorder=100,hatch='///')
    gdf_metric_threshold_basins.plot(edgecolor='k',facecolor='none',linewidth=0.2,ax=ax,zorder=101)
    gdf_climate_boundary.plot(ax=ax,linewidth=2,zorder=102,facecolor='none')
    # gdf_climate_boundary.loc[['O'],:].plot(ax=ax,linewidth=2,zorder=102,facecolor='none',edgecolor='k')
    # gdf_climate_boundary.loc[['S'],:].plot(ax=ax,linewidth=2,zorder=102,facecolor='none',edgecolor='r')
    # gdf_climate_boundary.loc[['W'],:].plot(ax=ax,linewidth=2,zorder=102,facecolor='none',edgecolor='b')

    # X-Labels
    if i==0:
        ax.set_title('(a) NNSE > 0.5',fontsize=axis_label_fontsize)
    if i==1:
        ax.set_title('(b) LNNSE > 0.5',fontsize=axis_label_fontsize)
    if i==2:
        ax.set_title('(c) |PBIAS| $\leq$ 50%',fontsize=axis_label_fontsize)

    # Y-Labels
    if i==0:
        #ax.set_ylabel('Hourly',fontsize=axis_label_fontsize)
        ax.text(-0.05,0.6,'S',rotation=90,fontsize=axis_label_fontsize,ha='center',va='center',transform=ax.transAxes,weight='bold')
    if i==3:
        # ax.set_ylabel('Daily',fontsize=axis_label_fontsize)
        ax.text(-0.05,0.6,'W',rotation=90,fontsize=axis_label_fontsize,ha='center',va='center',transform=ax.transAxes,weight='bold')

#     # Subplot labels
#     ax.text(0.1,0.95,subplot_labels[i],fontsize=subplot_label_fontsize,ha='center',va='center',transform=ax.transAxes)

    # Remove axis
    ax.axis('off')

    i+=1

# Cbar
# Make space for colorbar
fig.subplots_adjust(hspace=-1.0,wspace=-0.6)
plt.tight_layout()
fig.subplots_adjust(bottom=0.05)

cbar_width = 0.20
cbar_height = 0.02

# Central Cbar
cbar_ax = fig.add_axes([0.1+0.32, 0.1, cbar_width, cbar_height])
cb = mpl.colorbar.ColorbarBase(cmap=plot_prop['PBIAS']['cmap'],
                        norm=plot_prop['PBIAS']['norm'],
                        ticks=plot_prop['PBIAS']['bounds'],
                        ax=cbar_ax,
                        orientation='horizontal')
cb.ax.tick_params(labelsize=14)
cb.ax.set_title('Gauges (%)',y=1.02,fontsize=16)
cb.set_ticks([0,20,40,60,80,100])


# # NNSE
# cbar_ax = fig.add_axes([0.1, 0.1, cbar_width, cbar_height])
# cb = mpl.colorbar.ColorbarBase(cmap=plot_prop['NNSE']['cmap'],
#                         norm=plot_prop['NNSE']['norm'],
#                         ticks=plot_prop['NNSE']['bounds'],
#                         ax=cbar_ax,
#                         orientation='horizontal')
# cb.ax.tick_params(labelsize=14)
# cb.ax.set_title('Gauges (%)',y=1.02,fontsize=16)
# cb.set_ticks([0,20,40,60,80,100])

# # LNNSE
# cbar_ax = fig.add_axes([0.1+0.32, 0.1, cbar_width, cbar_height])
# cb = mpl.colorbar.ColorbarBase(cmap=plot_prop['LNNSE']['cmap'],
#                         norm=plot_prop['LNNSE']['norm'],
#                         ticks=plot_prop['LNNSE']['bounds'],
#                         ax=cbar_ax,
#                         orientation='horizontal')
# cb.ax.tick_params(labelsize=14)
# cb.ax.set_title('Gauges (%)',y=1.02,fontsize=16)
# if frequency == 'daily':
#     cb.set_ticks([0,20,40,60,70])
# if frequency == 'hourly':
#     cb.set_ticks([0,20,40,60])

# # PBIAS
# cbar_ax = fig.add_axes([0.1+0.32*2, 0.1, cbar_width, cbar_height])
# cb = mpl.colorbar.ColorbarBase(cmap=plot_prop['PBIAS']['cmap'],
#                         norm=plot_prop['PBIAS']['norm'],
#                         ticks=plot_prop['PBIAS']['bounds'],
#                         ax=cbar_ax,
#                         orientation='horizontal')
# cb.ax.tick_params(labelsize=14)
# cb.ax.set_title('Gauges (%)',y=1.02,fontsize=16)
# cb.set_ticks([0,20,40,60,80,100])

fig.savefig(os.path.join(savedir,f'{frequency}.pdf'),dpi=300,bbox_inches='tight')
fig.savefig(os.path.join(savedir,f'{frequency}.svg'),dpi=300,bbox_inches='tight')
fig.savefig(os.path.join(savedir,f'{frequency}.png'),dpi=300,bbox_inches='tight')








