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

axis_label_fontsize = 14
subplot_label_fontsize = 12
cbar_tick_label_fontsize = 12

frequency = 'daily'

cmap_prop = {
    'NNSE':{'bounds':np.array([0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9]),
            'norm':colors.BoundaryNorm(boundaries=np.array([0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9]), ncolors=256),
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

gp_metrics = os.path.join(metrics_dir,'{}_{}_{}_utm12n_nad83.parquet.gzip'.format(frequency,str(start_wy),str(end_wy)))
gdf_metrics = gpd.read_parquet(gp_metrics)

marker_dict = {'UG': 'o', 'SA': 's', 'LCO': '^', 'SJ': 'D','VE':'>'}  # Adjust this as needed
# Create a new column in the DataFrame with the marker styles
gdf_metrics['marker'] = gdf_metrics['M_BASIN_ABR'].map(marker_dict)

# SWE (RMSE vs PBIAS vs NNSE)
varx = 'SWE_RMSE'
vary = 'SWE_PBIAS'
varh = 'SWE_NNSE'
cmap = cmap_prop[varh.split('_')[-1]]['cmap']
norm = cmap_prop[varh.split('_')[-1]]['norm']
extend=get_cmap_extend(cmap_prop[varh.split('_')[-1]]['bounds'].min(),
                       cmap_prop[varh.split('_')[-1]]['bounds'].max(),
                       gdf_metrics[varh].min(),
                       gdf_metrics[varh].max())

fig,ax = plt.subplots(1,1,figsize=(5,3))
for marker in marker_dict.values():
    df_marker = gdf_metrics[gdf_metrics['marker'] == marker]
    sns.scatterplot(data=df_marker,
                    x=varx,y=vary,ax=ax,hue=varh,
                    hue_norm=cmap_prop[varh.split('_')[-1]]['norm'],
                    palette=cmap_prop[varh.split('_')[-1]]['cmap'],
                    marker=marker,
                    s=60, edgecolor='k',legend=False,lw=0.5)
ax.set_box_aspect(1)
ax.set_xlabel('RMSE (mm/day)',fontsize=axis_label_fontsize)
ax.set_ylabel('PBIAS (%)',fontsize=axis_label_fontsize)
ax.grid(True,color='gray',linestyle='--',linewidth=0.5)
ax.tick_params(axis='both', which='major', labelsize=cbar_tick_label_fontsize)
ax.axhline(0,color='k',linestyle='--',linewidth=1)

# Set x and y ticks at 20 interval
ax.set_xticks(range(int(ax.get_xlim()[0]) // 40 * 40, int(ax.get_xlim()[1]) // 40 * 40 + 1, 40))
ax.set_yticks(range(int(ax.get_ylim()[0]) // 20 * 20, int(ax.get_ylim()[1]) // 20 * 20 + 1, 20))


# Create a list of Line2D objects for the legend
legend_elements = [Line2D([0], [0], marker=marker, color='w', label=label, 
                          markerfacecolor='k', markersize=10) 
                   for label, marker in marker_dict.items()]

# Add the legend to the figure
# fig.legend(handles=legend_elements, loc='lower center', ncol=1, fontsize=12,bbox_to_anchor=(0.6, 0.65),frameon=False,prop={'size': 10})


plt.tight_layout()

# Create a ScalarMappable object for the colorbar
plt.subplots_adjust(right=0.8)
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])  # You can set an empty array for the ScalarMappable
# Add the colorbar to the figure
cbar = fig.colorbar(sm, ax=ax, orientation='vertical', pad=0.01)
cbar.set_label('NNSE', size=axis_label_fontsize)
cbar.ax.set_position([0.7, 0.2, 0.05, 0.7])
cbar.set_ticks(cmap_prop[varh.split('_')[-1]]['bounds'])
cbar.set_ticklabels(cmap_prop[varh.split('_')[-1]]['bounds'])
cbar.ax.tick_params(labelsize=cbar_tick_label_fontsize)

fig.savefig(os.path.join(savedir,'{}_{}_{}_{}_scatter.png'.format(frequency,varx,vary,varh)),dpi=300,bbox_inches='tight')
fig.savefig(os.path.join(savedir,'{}_{}_{}_{}_scatter.pdf'.format(frequency,varx,vary,varh)),dpi=300,bbox_inches='tight')
fig.savefig(os.path.join(savedir,'{}_{}_{}_{}_scatter.svg'.format(frequency,varx,vary,varh)),dpi=300,bbox_inches='tight')




# # SWE
# var_name = 'SWE'
# # DeltaP vs Elevation vs PBIAS
# metric_name = 'PBIAS'
# fig,ax = plt.subplots(1,1,figsize=(5,4))
# sns.scatterplot(data=gdf_metrics,x='SNOTEL_Elevation_m',y='Delta_PRECIP_A',
#                 hue=f'{var_name}_{metric_name}',hue_norm=cmap_prop[metric_name]['norm'],
#                 ax=ax,legend=False,palette=cmap_prop[metric_name]['cmap'],
#                 s=60, edgecolor='k',lw=0.5)
# ax.set_box_aspect(1)
# plt.tight_layout()
# fig.subplots_adjust(right=0.90)
# cbar_ax = fig.add_axes([0.90, 0.15, 0.03, 0.7])
# extend=get_cmap_extend(cmap_prop[metric_name]['bounds'].min(),
#                        cmap_prop[metric_name]['bounds'].max(),
#                        gdf_metrics[f'{var_name}_{metric_name}'].min(),
#                        gdf_metrics[f'{var_name}_{metric_name}'].max())
# cb = mpl.colorbar.ColorbarBase(cmap=cmap_prop[metric_name]['cmap'],
#                     norm=cmap_prop[metric_name]['norm'],
#                     ticks=cmap_prop[metric_name]['bounds'],
#                     ax=cbar_ax,extend=extend)
# cb.ax.tick_params(labelsize=cbar_tick_label_fontsize) 
# cb.ax.set_title(f'{var_name}\n'+metric_name+' (%)',y=1.02,fontsize=subplot_label_fontsize)
# ax.tick_params(axis='both', which='major', labelsize=subplot_label_fontsize)
# ax.set_xlabel('SNOTEL Elevation (m)',fontsize=axis_label_fontsize)
# ax.set_ylabel('AORC $\it{P}$ - SNOTEL $\it{P}$ (mm)',fontsize=axis_label_fontsize)
# ax.set_axisbelow(True)
# ax.xaxis.grid(color='gray', linestyle='dashed')
# ax.yaxis.grid(color='gray', linestyle='dashed')
# fig.savefig(os.path.join(savedir,'{}_{}_{}_DeltaP_E_PBIAS_scatter.png'.format(frequency,var_name,metric_name)),dpi=300,bbox_inches='tight')
# fig.savefig(os.path.join(savedir,'{}_{}_{}_DeltaP_E_PBIAS_scatter.pdf'.format(frequency,var_name,metric_name)),dpi=300,bbox_inches='tight')

# # DeltaP vs Elevation vs BIAS
# metric_name = 'BIAS'
# gdf_metrics[f'{var_name}_{metric_name}'] = gdf_metrics[f'{var_name}_{metric_name}']*1000
# fig,ax = plt.subplots(1,1,figsize=(5,4))
# sns.scatterplot(data=gdf_metrics,x='SNOTEL_Elevation_m',y='Delta_PRECIP_A',
#                 hue=f'{var_name}_{metric_name}',hue_norm=cmap_prop[metric_name]['norm'],
#                 ax=ax,legend=False,palette=cmap_prop[metric_name]['cmap'],
#                 s=60, edgecolor='k',lw=0.5)
# ax.set_box_aspect(1)
# plt.tight_layout()
# fig.subplots_adjust(right=0.90)
# cbar_ax = fig.add_axes([0.90, 0.15, 0.03, 0.7])
# extend=get_cmap_extend(cmap_prop[metric_name]['bounds'].min(),
#                        cmap_prop[metric_name]['bounds'].max(),
#                        gdf_metrics[f'{var_name}_{metric_name}'].min(),
#                        gdf_metrics[f'{var_name}_{metric_name}'].max())
# cb = mpl.colorbar.ColorbarBase(cmap=cmap_prop[metric_name]['cmap'],
#                     norm=cmap_prop[metric_name]['norm'],
#                     ticks=cmap_prop[metric_name]['bounds'],
#                     ax=cbar_ax,extend=extend)
# cb.ax.tick_params(labelsize=cbar_tick_label_fontsize) 
# cb.ax.set_title(f'{var_name}\n'+metric_name+' (mm)',y=1.02,fontsize=subplot_label_fontsize)
# ax.tick_params(axis='both', which='major', labelsize=subplot_label_fontsize)
# ax.set_xlabel('SNOTEL Elevation (m)',fontsize=axis_label_fontsize)
# ax.set_ylabel('AORC $\it{P}$ - SNOTEL $\it{P}$ (mm)',fontsize=axis_label_fontsize)
# ax.set_axisbelow(True)
# ax.xaxis.grid(color='gray', linestyle='dashed')
# ax.yaxis.grid(color='gray', linestyle='dashed')
# fig.savefig(os.path.join(savedir,'{}_{}_{}_DeltaP_E_BIAS_scatter.png'.format(frequency,var_name,metric_name)),dpi=300,bbox_inches='tight')
# fig.savefig(os.path.join(savedir,'{}_{}_{}_DeltaP_E_BIAS_scatter.pdf'.format(frequency,var_name,metric_name)),dpi=300,bbox_inches='tight')


# # NSE vs Elevation vs PBIAS
# metric_name = 'PBIAS'
# fig,ax = plt.subplots(1,1,figsize=(5,4))
# sns.scatterplot(data=gdf_metrics,x='SNOTEL_Elevation_m',y='SWE_NSE',
#                 hue=f'{var_name}_{metric_name}',hue_norm=cmap_prop[metric_name]['norm'],
#                 ax=ax,legend=False,palette=cmap_prop[metric_name]['cmap'],
#                 s=60, edgecolor='k',lw=0.5)
# ax.set_box_aspect(1)
# plt.tight_layout()
# fig.subplots_adjust(right=0.90)
# cbar_ax = fig.add_axes([0.90, 0.15, 0.03, 0.7])
# extend=get_cmap_extend(cmap_prop[metric_name]['bounds'].min(),
#                        cmap_prop[metric_name]['bounds'].max(),
#                        gdf_metrics[f'{var_name}_{metric_name}'].min(),
#                        gdf_metrics[f'{var_name}_{metric_name}'].max())
# cb = mpl.colorbar.ColorbarBase(cmap=cmap_prop[metric_name]['cmap'],
#                     norm=cmap_prop[metric_name]['norm'],
#                     ticks=cmap_prop[metric_name]['bounds'],
#                     ax=cbar_ax,extend=extend)
# cb.ax.tick_params(labelsize=cbar_tick_label_fontsize) 
# cb.ax.set_title(f'{var_name}\n'+metric_name+' (%)',y=1.02,fontsize=subplot_label_fontsize)
# ax.tick_params(axis='both', which='major', labelsize=subplot_label_fontsize)
# ax.set_xlabel('SNOTEL Elevation (m)',fontsize=axis_label_fontsize)
# ax.set_ylabel('NSE',fontsize=axis_label_fontsize)
# ax.set_axisbelow(True)
# ax.xaxis.grid(color='gray', linestyle='dashed')
# ax.yaxis.grid(color='gray', linestyle='dashed')
# fig.savefig(os.path.join(savedir,'{}_{}_{}_NSE_E_PBIAS_scatter.png'.format(frequency,var_name,metric_name)),dpi=300,bbox_inches='tight')
# fig.savefig(os.path.join(savedir,'{}_{}_{}_NSE_E_PBIAS_scatter.pdf'.format(frequency,var_name,metric_name)),dpi=300,bbox_inches='tight')



# # SNOWD
# var_name = 'SNOWD'
# # DeltaP vs Elevation vs PBIAS
# metric_name = 'PBIAS'
# fig,ax = plt.subplots(1,1,figsize=(5,4))
# sns.scatterplot(data=gdf_metrics,x='SNOTEL_Elevation_m',y='Delta_PRECIP_A',
#                 hue=f'{var_name}_{metric_name}',hue_norm=cmap_prop[metric_name]['norm'],
#                 ax=ax,legend=False,palette=cmap_prop[metric_name]['cmap'],
#                 s=60, edgecolor='k',lw=0.5)
# ax.set_box_aspect(1)
# plt.tight_layout()
# fig.subplots_adjust(right=0.90)
# cbar_ax = fig.add_axes([0.90, 0.15, 0.03, 0.7])
# extend=get_cmap_extend(cmap_prop[metric_name]['bounds'].min(),
#                        cmap_prop[metric_name]['bounds'].max(),
#                        gdf_metrics[f'{var_name}_{metric_name}'].min(),
#                        gdf_metrics[f'{var_name}_{metric_name}'].max())
# cb = mpl.colorbar.ColorbarBase(cmap=cmap_prop[metric_name]['cmap'],
#                     norm=cmap_prop[metric_name]['norm'],
#                     ticks=cmap_prop[metric_name]['bounds'],
#                     ax=cbar_ax,extend=extend)
# cb.ax.tick_params(labelsize=cbar_tick_label_fontsize) 
# cb.ax.set_title(f'{var_name}\n'+metric_name+' (%)',y=1.02,fontsize=subplot_label_fontsize)
# ax.tick_params(axis='both', which='major', labelsize=subplot_label_fontsize)
# ax.set_xlabel('SNOTEL Elevation (m)',fontsize=axis_label_fontsize)
# ax.set_ylabel('AORC $\it{P}$ - SNOTEL $\it{P}$ (mm)',fontsize=axis_label_fontsize)
# ax.set_axisbelow(True)
# ax.xaxis.grid(color='gray', linestyle='dashed')
# ax.yaxis.grid(color='gray', linestyle='dashed')
# fig.savefig(os.path.join(savedir,'{}_{}_{}_DeltaP_E_PBIAS_scatter.png'.format(frequency,var_name,metric_name)),dpi=300,bbox_inches='tight')
# fig.savefig(os.path.join(savedir,'{}_{}_{}_DeltaP_E_PBIAS_scatter.pdf'.format(frequency,var_name,metric_name)),dpi=300,bbox_inches='tight')

# # NSE vs Elevation vs PBIAS
# metric_name = 'PBIAS'
# fig,ax = plt.subplots(1,1,figsize=(5,4))
# sns.scatterplot(data=gdf_metrics,x='SNOTEL_Elevation_m',y='SWE_NSE',
#                 hue=f'{var_name}_{metric_name}',hue_norm=cmap_prop[metric_name]['norm'],
#                 ax=ax,legend=False,palette=cmap_prop[metric_name]['cmap'],
#                 s=60, edgecolor='k',lw=0.5)
# ax.set_box_aspect(1)
# plt.tight_layout()
# fig.subplots_adjust(right=0.90)
# cbar_ax = fig.add_axes([0.90, 0.15, 0.03, 0.7])
# extend=get_cmap_extend(cmap_prop[metric_name]['bounds'].min(),
#                        cmap_prop[metric_name]['bounds'].max(),
#                        gdf_metrics[f'{var_name}_{metric_name}'].min(),
#                        gdf_metrics[f'{var_name}_{metric_name}'].max())
# cb = mpl.colorbar.ColorbarBase(cmap=cmap_prop[metric_name]['cmap'],
#                     norm=cmap_prop[metric_name]['norm'],
#                     ticks=cmap_prop[metric_name]['bounds'],
#                     ax=cbar_ax,extend=extend)
# cb.ax.tick_params(labelsize=cbar_tick_label_fontsize) 
# cb.ax.set_title(f'{var_name}\n'+metric_name+' (%)',y=1.02,fontsize=subplot_label_fontsize)
# ax.tick_params(axis='both', which='major', labelsize=subplot_label_fontsize)
# ax.set_xlabel('SNOTEL Elevation (m)',fontsize=axis_label_fontsize)
# ax.set_ylabel('NSE',fontsize=axis_label_fontsize)
# ax.set_axisbelow(True)
# ax.xaxis.grid(color='gray', linestyle='dashed')
# ax.yaxis.grid(color='gray', linestyle='dashed')
# fig.savefig(os.path.join(savedir,'{}_{}_{}_NSE_E_PBIAS_scatter.png'.format(frequency,var_name,metric_name)),dpi=300,bbox_inches='tight')
# fig.savefig(os.path.join(savedir,'{}_{}_{}_NSE_E_PBIAS_scatter.pdf'.format(frequency,var_name,metric_name)),dpi=300,bbox_inches='tight')
