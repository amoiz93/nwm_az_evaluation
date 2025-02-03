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

import custom_basemaps as cbm

import param_nwm3
import misc

def get_cmap_extend(min_bound,max_bound,min_value,max_value):
    extend = 'neither'
    if min_value < min_bound:
        extend = 'min'
    if max_value > max_bound:
        extend = 'max'
    if min_value < min_bound and max_value > max_bound:
        extend = 'both'
    return extend

savedir = os.path.join(param_nwm3.out_dir,'figures',os.path.basename(__file__).split('.')[0])
misc.makedir(savedir)

metrics_dir = '../out/02_calc_metrics'
start_wy = 2003
end_wy = 2022

axis_label_fontsize = 22
subplot_label_fontsize = 16
cbar_tick_label_fontsize = 12

# CMAP Properties
cmap_prop = {
    'NNSE':{'bounds':np.array([0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]),
            'norm':colors.BoundaryNorm(boundaries=np.array([0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]), ncolors=256),
            'cmap':'turbo_r'},

    'NSE':{'bounds':np.array([0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]),
            'norm':colors.BoundaryNorm(boundaries=np.array([0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]), ncolors=256),
            'cmap':'turbo_r'},

    'PBIAS':{'bounds':np.array([-100,-80,-60,-40,-20,0,20,40,60,80,100]),
            'norm':colors.BoundaryNorm(boundaries=np.array([-100,-80,-60,-40,-20,0,20,40,60,80,100]), ncolors=256),
            'cmap':'RdBu'},

    'BIAS':{'bounds':np.array([-50,-40,-30,-20,-10,0,10,20,30,40,50]),
            'norm':colors.BoundaryNorm(boundaries=np.array([-50,-40,-30,-20,-10,0,10,20,30,40,50]), ncolors=256),
            'cmap':'RdBu'},
}


var_name = 'SWE'

# NNSE
fig,axs = plt.subplots(1,3,subplot_kw={'projection':param_nwm3.cartopy_crs_atur_utm12n},figsize=(8,8))
metric_name = 'NNSE'
i=0
for ax in axs.reshape(-1):
    ax=cbm.az_huc8_basemap(ax=ax,gridlines_flag=False)

    if i==0:
        pq_metrics = os.path.join(metrics_dir,f'{ "hourly" }_{str(start_wy)}_{str(end_wy)}_utm12n_nad83.parquet.gzip')
        gdf_metrics = gpd.read_parquet(pq_metrics)
        gdf_metrics.plot(column=f'{var_name}_{metric_name}',ax=ax,cmap=cmap_prop[metric_name]['cmap'],norm=cmap_prop[metric_name]['norm'],legend=False,zorder=100,markersize=20)
        ax.set_title('Hourly',fontsize=subplot_label_fontsize)
    if i==1:
        pq_metrics = os.path.join(metrics_dir,f'{ "daily" }_{str(start_wy)}_{str(end_wy)}_utm12n_nad83.parquet.gzip')
        gdf_metrics = gpd.read_parquet(pq_metrics)
        gdf_metrics.plot(column=f'{var_name}_{metric_name}',ax=ax,cmap=cmap_prop[metric_name]['cmap'],norm=cmap_prop[metric_name]['norm'],legend=False,zorder=100,markersize=20)
        ax.set_title('Daily',fontsize=subplot_label_fontsize)
    if i==2:
        pq_metrics = os.path.join(metrics_dir,f'{ "monthly" }_{str(start_wy)}_{str(end_wy)}_utm12n_nad83.parquet.gzip')
        gdf_metrics = gpd.read_parquet(pq_metrics)
        gdf_metrics.plot(column=f'{var_name}_{metric_name}',ax=ax,cmap=cmap_prop[metric_name]['cmap'],norm=cmap_prop[metric_name]['norm'],legend=False,zorder=100,markersize=20)
        ax.set_title('Monthly',fontsize=subplot_label_fontsize)
    ax.axis('off')
    i+=1

fig.subplots_adjust(hspace=-0.6,wspace=-0.3)
plt.tight_layout()
fig.subplots_adjust(right=0.90)
cbar_ax = fig.add_axes([0.95, 0.35, 0.02, 0.3])
cb = mpl.colorbar.ColorbarBase(cmap=cmap_prop[metric_name]['cmap'],
                    norm=cmap_prop[metric_name]['norm'],
                    ticks=cmap_prop[metric_name]['bounds'],
                    ax=cbar_ax)
cb.ax.set_title(f'{var_name}\n'+metric_name,y=1.02,fontsize=subplot_label_fontsize)
cb.ax.tick_params(labelsize=cbar_tick_label_fontsize) 
fig.savefig(os.path.join(savedir,f'{var_name}_{metric_name}.png'),dpi=300,bbox_inches='tight')
fig.savefig(os.path.join(savedir,f'{var_name}_{metric_name}.pdf'),dpi=300,bbox_inches='tight')


# NSE
fig,axs = plt.subplots(1,3,subplot_kw={'projection':param_nwm3.cartopy_crs_atur_utm12n},figsize=(8,8))
metric_name = 'NSE'
i=0
for ax in axs.reshape(-1):
    ax=cbm.az_huc8_basemap(ax=ax,gridlines_flag=False)

    if i==0:
        pq_metrics = os.path.join(metrics_dir,f'{ "hourly" }_{str(start_wy)}_{str(end_wy)}_utm12n_nad83.parquet.gzip')
        gdf_metrics = gpd.read_parquet(pq_metrics)
        gdf_metrics.plot(column=f'{var_name}_{metric_name}',ax=ax,cmap=cmap_prop[metric_name]['cmap'],norm=cmap_prop[metric_name]['norm'],legend=False,zorder=100,markersize=20)
        ax.set_title('Hourly',fontsize=subplot_label_fontsize)
    if i==1:
        pq_metrics = os.path.join(metrics_dir,f'{ "daily" }_{str(start_wy)}_{str(end_wy)}_utm12n_nad83.parquet.gzip')
        gdf_metrics = gpd.read_parquet(pq_metrics)
        gdf_metrics.plot(column=f'{var_name}_{metric_name}',ax=ax,cmap=cmap_prop[metric_name]['cmap'],norm=cmap_prop[metric_name]['norm'],legend=False,zorder=100,markersize=20)
        ax.set_title('Daily',fontsize=subplot_label_fontsize)
    if i==2:
        pq_metrics = os.path.join(metrics_dir,f'{ "monthly" }_{str(start_wy)}_{str(end_wy)}_utm12n_nad83.parquet.gzip')
        gdf_metrics = gpd.read_parquet(pq_metrics)
        gdf_metrics.plot(column=f'{var_name}_{metric_name}',ax=ax,cmap=cmap_prop[metric_name]['cmap'],norm=cmap_prop[metric_name]['norm'],legend=False,zorder=100,markersize=20)
        ax.set_title('Monthly',fontsize=subplot_label_fontsize)
    ax.axis('off')
    i+=1

fig.subplots_adjust(hspace=-0.6,wspace=-0.3)
plt.tight_layout()
fig.subplots_adjust(right=0.90)
cbar_ax = fig.add_axes([0.95, 0.35, 0.02, 0.3])
cb = mpl.colorbar.ColorbarBase(cmap=cmap_prop[metric_name]['cmap'],
                    norm=cmap_prop[metric_name]['norm'],
                    ticks=cmap_prop[metric_name]['bounds'],
                    ax=cbar_ax)
cb.ax.set_title(f'{var_name}\n'+metric_name,y=1.02,fontsize=subplot_label_fontsize)
cb.ax.tick_params(labelsize=cbar_tick_label_fontsize) 
fig.savefig(os.path.join(savedir,f'{var_name}_{metric_name}.png'),dpi=300,bbox_inches='tight')
fig.savefig(os.path.join(savedir,f'{var_name}_{metric_name}.pdf'),dpi=300,bbox_inches='tight')


# PBIAS
fig,axs = plt.subplots(1,3,subplot_kw={'projection':param_nwm3.cartopy_crs_atur_utm12n},figsize=(8,8))
metric_name = 'PBIAS'
i=0
for ax in axs.reshape(-1):
    ax=cbm.az_huc8_basemap(ax=ax,gridlines_flag=False)

    if i==0:
        pq_metrics = os.path.join(metrics_dir,f'{ "hourly" }_{str(start_wy)}_{str(end_wy)}_utm12n_nad83.parquet.gzip')
        gdf_metrics = gpd.read_parquet(pq_metrics)
        gdf_metrics.plot(column=f'{var_name}_{metric_name}',ax=ax,cmap=cmap_prop[metric_name]['cmap'],norm=cmap_prop[metric_name]['norm'],legend=False,zorder=100,markersize=20)
        ax.set_title('Hourly',fontsize=subplot_label_fontsize)
    if i==1:
        pq_metrics = os.path.join(metrics_dir,f'{ "daily" }_{str(start_wy)}_{str(end_wy)}_utm12n_nad83.parquet.gzip')
        gdf_metrics = gpd.read_parquet(pq_metrics)
        gdf_metrics.plot(column=f'{var_name}_{metric_name}',ax=ax,cmap=cmap_prop[metric_name]['cmap'],norm=cmap_prop[metric_name]['norm'],legend=False,zorder=100,markersize=20)
        ax.set_title('Daily',fontsize=subplot_label_fontsize)
    if i==2:
        pq_metrics = os.path.join(metrics_dir,f'{ "monthly" }_{str(start_wy)}_{str(end_wy)}_utm12n_nad83.parquet.gzip')
        gdf_metrics = gpd.read_parquet(pq_metrics)
        gdf_metrics.plot(column=f'{var_name}_{metric_name}',ax=ax,cmap=cmap_prop[metric_name]['cmap'],norm=cmap_prop[metric_name]['norm'],legend=False,zorder=100,markersize=20)
        ax.set_title('Monthly',fontsize=subplot_label_fontsize)
    ax.axis('off')
    i+=1

fig.subplots_adjust(hspace=-0.6,wspace=-0.3)
plt.tight_layout()
fig.subplots_adjust(right=0.90)
cbar_ax = fig.add_axes([0.95, 0.35, 0.02, 0.3])
cb = mpl.colorbar.ColorbarBase(cmap=cmap_prop[metric_name]['cmap'],
                    norm=cmap_prop[metric_name]['norm'],
                    ticks=cmap_prop[metric_name]['bounds'],
                    ax=cbar_ax)
cb.ax.tick_params(labelsize=cbar_tick_label_fontsize) 
cb.ax.set_title(f'{var_name}\n'+metric_name+' (%)',y=1.02,fontsize=subplot_label_fontsize)
fig.savefig(os.path.join(savedir,f'{var_name}_{metric_name}.png'),dpi=300,bbox_inches='tight')
fig.savefig(os.path.join(savedir,f'{var_name}_{metric_name}.pdf'),dpi=300,bbox_inches='tight')


# BIAS
# fig,axs = plt.subplots(1,3,subplot_kw={'projection':param_nwm3.cartopy_crs_atur_utm12n},figsize=(8,8))
metric_name = 'BIAS'
i=0
for freq in ['hourly','daily','monthly']:
# for ax in axs.reshape(-1):
#     ax=cbm.az_huc8_basemap(ax=ax,gridlines_flag=False)

    # if i==0:
    fig,ax = plt.subplots(1,1,subplot_kw={'projection':param_nwm3.cartopy_crs_atur_utm12n},figsize=(4,4))
    ax=cbm.az_huc8_basemap(ax=ax,gridlines_flag=False)
    pq_metrics = os.path.join(metrics_dir,f'{ freq }_{str(start_wy)}_{str(end_wy)}_utm12n_nad83.parquet.gzip')
    gdf_metrics = gpd.read_parquet(pq_metrics)
    gdf_metrics[f'{var_name}_{metric_name}'] = gdf_metrics[f'{var_name}_{metric_name}']*1000
    gdf_metrics.plot(column=f'{var_name}_{metric_name}',ax=ax,cmap=cmap_prop[metric_name]['cmap'],norm=cmap_prop[metric_name]['norm'],legend=False,zorder=100,markersize=30)
    ax.set_title(freq.capitalize(),fontsize=subplot_label_fontsize)
    ax.axis('off')
    plt.tight_layout()
    fig.subplots_adjust(right=0.90)
    cbar_ax = fig.add_axes([0.95, 0.1, 0.04, 0.7])
    extend=get_cmap_extend(cmap_prop[metric_name]['bounds'].min(),
                    cmap_prop[metric_name]['bounds'].max(),
                    gdf_metrics[f'{var_name}_{metric_name}'].min(),
                    gdf_metrics[f'{var_name}_{metric_name}'].max())
    cb = mpl.colorbar.ColorbarBase(cmap=cmap_prop[metric_name]['cmap'],
                norm=cmap_prop[metric_name]['norm'],
                ticks=cmap_prop[metric_name]['bounds'],
                ax=cbar_ax,extend=extend)
    cb.ax.tick_params(labelsize=cbar_tick_label_fontsize)
    cb.ax.set_title(f'{var_name}\nMean\n'+metric_name+' (mm)',y=1.02,fontsize=subplot_label_fontsize)
    fig.savefig(os.path.join(savedir,f'{var_name}_{metric_name}_{freq}.png'),dpi=300,bbox_inches='tight')
    fig.savefig(os.path.join(savedir,f'{var_name}_{metric_name}_{freq}.pdf'),dpi=300,bbox_inches='tight')

    # if i==1:
    #     pq_metrics = os.path.join(metrics_dir,f'{ "daily" }_{str(start_wy)}_{str(end_wy)}_utm12n_nad83.parquet.gzip')
    #     gdf_metrics = gpd.read_parquet(pq_metrics)
    #     gdf_metrics[f'{var_name}_{metric_name}'] = gdf_metrics[f'{var_name}_{metric_name}']*1000
    #     gdf_metrics.plot(column=f'{var_name}_{metric_name}',ax=ax,cmap=cmap_prop[metric_name]['cmap'],norm=cmap_prop[metric_name]['norm'],legend=False,zorder=100,markersize=20)
    #     ax.set_title('Daily',fontsize=subplot_label_fontsize)
    # if i==2:
    #     pq_metrics = os.path.join(metrics_dir,f'{ "monthly" }_{str(start_wy)}_{str(end_wy)}_utm12n_nad83.parquet.gzip')
    #     gdf_metrics = gpd.read_parquet(pq_metrics)
    #     gdf_metrics[f'{var_name}_{metric_name}'] = gdf_metrics[f'{var_name}_{metric_name}']*1000
    #     gdf_metrics.plot(column=f'{var_name}_{metric_name}',ax=ax,cmap=cmap_prop[metric_name]['cmap'],norm=cmap_prop[metric_name]['norm'],legend=False,zorder=100,markersize=20)
    #     ax.set_title('Monthly',fontsize=subplot_label_fontsize)
    # ax.axis('off')
    i+=1

# fig.subplots_adjust(hspace=-0.6,wspace=-0.3)
# plt.tight_layout()
# fig.subplots_adjust(right=0.90)
# cbar_ax = fig.add_axes([0.95, 0.35, 0.02, 0.3])
# cb = mpl.colorbar.ColorbarBase(cmap=cmap_prop[metric_name]['cmap'],
#                     norm=cmap_prop[metric_name]['norm'],
#                     ticks=cmap_prop[metric_name]['bounds'],
#                     ax=cbar_ax)
# cb.ax.tick_params(labelsize=cbar_tick_label_fontsize) 
# cb.ax.set_title(f'{var_name}\n'+metric_name+' mm',y=1.02,fontsize=subplot_label_fontsize)
# fig.savefig(os.path.join(savedir,f'{var_name}_{metric_name}.png'),dpi=300,bbox_inches='tight')
# fig.savefig(os.path.join(savedir,f'{var_name}_{metric_name}.pdf'),dpi=300,bbox_inches='tight')
