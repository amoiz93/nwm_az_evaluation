import os
import sys
sys.path.append('../src')
import glob

import xarray as xr
import numpy as np
import math
import pandas as pd
import geopandas as gpd

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D
import seaborn as sns

import param_nwm3
import misc

mpl.rcParams['pdf.fonttype'] = 42


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

frequency = 'daily'

gp_metrics = os.path.join(metrics_dir,'{}_{}_{}_utm12n_nad83.parquet.gzip'.format(frequency,str(start_wy),str(end_wy)))
gdf_metrics = gpd.read_parquet(gp_metrics)

marker_dict = {'UG': 'o', 'SA': 's', 'LCO': '^', 'SJ': 'D','VE':'>'}  # Adjust this as needed
# Create a new column in the DataFrame with the marker styles
gdf_metrics['marker'] = gdf_metrics['M_BASIN_ABR'].map(marker_dict)


axis_label_fontsize = 14

fig,axs = plt.subplots(1,3,figsize=(10,3),sharex=True,sharey=False)
i=0
for ax in axs:
    # ax.set_aspect('equal')
    ax.set_box_aspect(1)
    varx = 'SNOTEL_Elevation_m'
    varh = 'Delta_SNOWFALL_WY'
    bounds = np.arange(-260,-20,20)
    norm = colors.BoundaryNorm(boundaries=bounds, ncolors=256)
    cmap = 'Reds_r'
    ax.set_xlabel('Elevation (m)',fontsize=axis_label_fontsize)
    if i==0:
        for marker in marker_dict.values():
            df_marker = gdf_metrics[gdf_metrics['marker'] == marker]
            vary = 'SWE_RMSE'
            ax.scatter(df_marker[varx],df_marker[vary],c=df_marker[varh],
            cmap=cmap,norm=norm,s=50,edgecolor='k',lw=0.5,marker=marker)
            ax.set_ylabel('RMSE (mm)',fontsize=axis_label_fontsize)
        
    if i==1:
        for marker in marker_dict.values():
            df_marker = gdf_metrics[gdf_metrics['marker'] == marker]
            vary = 'SWE_PBIAS'
            ax.axhline(0,color='k',linestyle='--')
            ax.scatter(df_marker[varx],df_marker[vary],c=df_marker[varh],
            cmap=cmap,norm=norm,s=50,edgecolor='k',lw=0.5,marker=marker)
            ax.set_ylabel('PBIAS (%)',fontsize=axis_label_fontsize)
    
    if i==2:
        for marker in marker_dict.values():
            df_marker = gdf_metrics[gdf_metrics['marker'] == marker]
            vary = 'SWE_NNSE'
            ax.axhline(0.5,color='k',linestyle='--')
            ax.scatter(df_marker[varx],df_marker[vary],c=df_marker[varh],
            cmap=cmap,norm=norm,s=50,edgecolor='k',lw=0.5,marker=marker)
            ax.set_ylabel('NNSE',fontsize=axis_label_fontsize)
        
    ax.grid(True,color='gray',linestyle='--',linewidth=0.5)
    ax.tick_params(axis='both', which='major', labelsize=12)

    
    # Set major and minor ticks
    ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=5))
    # ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=7))
    ax.text(0.02, 0.98, f'({chr(97 + i)})', transform=ax.transAxes, verticalalignment='top', fontsize=14)
    i+=1

# Create a ScalarMappable object for the colorbar
plt.subplots_adjust(right=0.8)
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])  # You can set an empty array for the ScalarMappable
# Add the colorbar to the figure
cbar = fig.colorbar(sm, ax=axs, orientation='vertical', pad=0.01)
# cbar.set_label('NWM - SNOTEL (mm)', size=axis_label_fontsize)
cbar.set_label('$\Delta P_{Snowfall}$ (mm)', size=axis_label_fontsize)
cbar.ax.set_position([1.0, 0.2, 0.05, 0.68])
cbar.set_ticks(bounds)
cbar.set_ticklabels(bounds)
cbar.ax.tick_params(labelsize=12)


# Create a list of Line2D objects for the legend
legend_elements = [Line2D([0], [0], marker=marker, color='w', label=label, 
                          markerfacecolor='k', markersize=10) 
                   for label, marker in marker_dict.items()]

# Add the legend to the figure
fig.legend(handles=legend_elements, loc='lower center', ncol=len(marker_dict), fontsize=12,bbox_to_anchor=(0.55, -0.16),frameon=False,prop={'size': 14})

plt.tight_layout()
fig.savefig(os.path.join(savedir,'SWE_metrics_elevation_snowfall.png'),dpi=300,bbox_inches='tight')
fig.savefig(os.path.join(savedir,'SWE_metrics_elevation_snowfall.pdf'),dpi=300,bbox_inches='tight')
fig.savefig(os.path.join(savedir,'SWE_metrics_elevation_snowfall.svg'),dpi=300,bbox_inches='tight')

fig,axs = plt.subplots(1,3,figsize=(10,3),sharex=True,sharey=False)
i=0
for ax in axs:
    # ax.set_aspect('equal')
    ax.set_box_aspect(1)
    varx = 'SNOTEL_Elevation_m'
    varh = 'Delta_RAINFALL_WY'
    bounds = np.arange(-120,90,20)
    norm = colors.BoundaryNorm(boundaries=bounds, ncolors=256)
    cmap = 'turbo_r'
    ax.set_xlabel('Elevation (m)',fontsize=axis_label_fontsize)
    if i==0:
        for marker in marker_dict.values():
            df_marker = gdf_metrics[gdf_metrics['marker'] == marker]
            vary = 'SWE_RMSE'
            ax.scatter(df_marker[varx],df_marker[vary],c=df_marker[varh],
            cmap=cmap,norm=norm,s=50,edgecolor='k',lw=0.5,marker=marker)
            ax.set_ylabel('RMSE (mm)',fontsize=axis_label_fontsize)
        
    if i==1:
        for marker in marker_dict.values():
            df_marker = gdf_metrics[gdf_metrics['marker'] == marker]
            vary = 'SWE_PBIAS'
            ax.axhline(0,color='k',linestyle='--')
            ax.scatter(df_marker[varx],df_marker[vary],c=df_marker[varh],
            cmap=cmap,norm=norm,s=50,edgecolor='k',lw=0.5,marker=marker)
            ax.set_ylabel('PBIAS (%)',fontsize=axis_label_fontsize)
        
    if i==2:
        for marker in marker_dict.values():
            df_marker = gdf_metrics[gdf_metrics['marker'] == marker]
            vary = 'SWE_NNSE'
            ax.axhline(0.5,color='k',linestyle='--')
            ax.scatter(df_marker[varx],df_marker[vary],c=df_marker[varh],
            cmap=cmap,norm=norm,s=50,edgecolor='k',lw=0.5,marker=marker)
            ax.set_ylabel('NNSE',fontsize=axis_label_fontsize)
        
    ax.grid(True,color='gray',linestyle='--',linewidth=0.5)
    ax.tick_params(axis='both', which='major', labelsize=12)

    
    # Set major and minor ticks
    ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=5))
    # ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=7))
    ax.text(0.02, 0.98, f'({chr(97 + i)})', transform=ax.transAxes, verticalalignment='top', fontsize=14)
    i+=1

# Create a ScalarMappable object for the colorbar
plt.subplots_adjust(right=0.8)
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])  # You can set an empty array for the ScalarMappable
# Add the colorbar to the figure
cbar = fig.colorbar(sm, ax=axs, orientation='vertical', pad=0.01)
# cbar.set_label('NWM - SNOTEL (mm)', size=axis_label_fontsize)
cbar.set_label('$\Delta P_{Rainfall}$ (mm)', size=axis_label_fontsize)
cbar.ax.set_position([1.0, 0.2, 0.05, 0.68])
cbar.set_ticks(bounds)
cbar.set_ticklabels(bounds)
cbar.ax.tick_params(labelsize=12)

# Create a list of Line2D objects for the legend
legend_elements = [Line2D([0], [0], marker=marker, color='w', label=label, 
                          markerfacecolor='k', markersize=10) 
                   for label, marker in marker_dict.items()]

# Add the legend to the figure
fig.legend(handles=legend_elements, loc='lower center', ncol=len(marker_dict), fontsize=12,bbox_to_anchor=(0.55, -0.16),frameon=False,prop={'size': 14})


plt.tight_layout()
fig.savefig(os.path.join(savedir,'SWE_metrics_elevation_rainfall.png'),dpi=300,bbox_inches='tight')
fig.savefig(os.path.join(savedir,'SWE_metrics_elevation_rainfall.pdf'),dpi=300,bbox_inches='tight')
fig.savefig(os.path.join(savedir,'SWE_metrics_elevation_rainfall.svg'),dpi=300,bbox_inches='tight')

fig,axs = plt.subplots(1,3,figsize=(10,3),sharex=True,sharey=False)
i=0
for ax in axs:
    # ax.set_aspect('equal')
    ax.set_box_aspect(1)
    varx = 'SNOTEL_Elevation_m'
    varh = 'Delta_T2D_WY'
    bounds = np.arange(0.5,6.5,0.5)
    norm = colors.BoundaryNorm(boundaries=bounds, ncolors=256)
    cmap = 'turbo_r'
    ax.set_xlabel('Elevation (m)',fontsize=axis_label_fontsize)
    if i==0:
        for marker in marker_dict.values():
            df_marker = gdf_metrics[gdf_metrics['marker'] == marker]
            vary = 'SWE_RMSE'
            ax.scatter(df_marker[varx],df_marker[vary],c=df_marker[varh],
            cmap=cmap,norm=norm,s=50,edgecolor='k',lw=0.5,marker=marker)
            ax.set_ylabel('RMSE (mm)',fontsize=axis_label_fontsize)
        
    if i==1:
        for marker in marker_dict.values():
            df_marker = gdf_metrics[gdf_metrics['marker'] == marker]
            vary = 'SWE_PBIAS'
            ax.axhline(0,color='k',linestyle='--')
            ax.scatter(df_marker[varx],df_marker[vary],c=df_marker[varh],
            cmap=cmap,norm=norm,s=50,edgecolor='k',lw=0.5,marker=marker)
            ax.set_ylabel('PBIAS (%)',fontsize=axis_label_fontsize)
        
    if i==2:
        for marker in marker_dict.values():
            df_marker = gdf_metrics[gdf_metrics['marker'] == marker]
            vary = 'SWE_NNSE'
            ax.axhline(0.5,color='k',linestyle='--')
            ax.scatter(df_marker[varx],df_marker[vary],c=df_marker[varh],
            cmap=cmap,norm=norm,s=50,edgecolor='k',lw=0.5,marker=marker)
            ax.set_ylabel('NNSE',fontsize=axis_label_fontsize)
        
    ax.grid(True,color='gray',linestyle='--',linewidth=0.5)
    ax.tick_params(axis='both', which='major', labelsize=12)

    
    # Set major and minor ticks
    ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=5))
    # ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=7))
    ax.text(0.02, 0.98, f'({chr(97 + i)})', transform=ax.transAxes, verticalalignment='top', fontsize=14)
    i+=1

# Create a ScalarMappable object for the colorbar
plt.subplots_adjust(right=0.8)
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])  # You can set an empty array for the ScalarMappable
# Add the colorbar to the figure
cbar = fig.colorbar(sm, ax=axs, orientation='vertical', pad=0.01)
# cbar.set_label('NWM - SNOTEL (mm)', size=axis_label_fontsize)
cbar.set_label('$\Delta T$ ($^o$C)', size=axis_label_fontsize)
cbar.ax.set_position([1.0, 0.2, 0.05, 0.68])
cbar.set_ticks(bounds)
cbar.set_ticklabels(bounds)
cbar.ax.tick_params(labelsize=12)

# Create a list of Line2D objects for the legend
legend_elements = [Line2D([0], [0], marker=marker, color='w', label=label, 
                          markerfacecolor='k', markersize=10) 
                   for label, marker in marker_dict.items()]

# Add the legend to the figure
fig.legend(handles=legend_elements, loc='lower center', ncol=len(marker_dict), fontsize=12,bbox_to_anchor=(0.55, -0.16),frameon=False,prop={'size': 14})


plt.tight_layout()
fig.savefig(os.path.join(savedir,'SWE_metrics_elevation_temperature.png'),dpi=300,bbox_inches='tight')
fig.savefig(os.path.join(savedir,'SWE_metrics_elevation_temperature.pdf'),dpi=300,bbox_inches='tight')
fig.savefig(os.path.join(savedir,'SWE_metrics_elevation_temperature.svg'),dpi=300,bbox_inches='tight')
