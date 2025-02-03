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
import matplotlib.ticker as ticker
import seaborn as sns

import custom_basemaps as cbm

import param_nwm3
import misc


savedir = os.path.join(param_nwm3.out_dir,'figures',os.path.basename(__file__).split('.')[0])
misc.makedir(savedir)

metrics_dir = '/Users/amoiz2/Servers/dropbox_asu/Personal/Work/20230807_ASU_SSEBE_Mascaro/Workspace/projects/20240311_USGS_NWM_Q_Validation_AZ/out/06A_add_attributes/parquet'
start_year = 2003
end_year = 2022
frequency = 'daily'

pq_metrics = os.path.join(metrics_dir,'{}_{}_{}_utm12n_nad83.parquet.gzip'.format(frequency,str(start_year),str(end_year)))
gdf_metrics = gpd.read_parquet(pq_metrics)

cmap_prop = {
    'NNSE':{'bounds':np.array([0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]),
            'norm':colors.BoundaryNorm(boundaries=np.array([0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]), ncolors=256),
            'cmap':'turbo_r'},

    'NSE':{'bounds':np.array([0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]),
            'norm':colors.BoundaryNorm(boundaries=np.array([0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]), ncolors=256),
            'cmap':'turbo_r'},
            
    'PBIAS':{'bounds':np.array([-200,-180,-160,-140,-120,-100,-80,-60,-40,-20,0,20,40,60,80,100,120,140,160,180,200]),
            'norm':colors.BoundaryNorm(boundaries=np.array([-200,-180,-160,-140,-120,-100,-80,-60,-40,-20,0,20,40,60,80,100,120,140,160,180,200]), ncolors=256),
            'cmap':'RdYlBu'},

    'BIAS':{'bounds':np.array([-50,-40,-30,-20,-10,0,10,20,30,40,50]),
            'norm':colors.BoundaryNorm(boundaries=np.array([-50,-40,-30,-20,-10,0,10,20,30,40,50]), ncolors=256),
            'cmap':'RdYlBu'},
}

# Metrics Scatter Plot
fig,ax = plt.subplots(1,3,figsize=(10,10),sharex=True,sharey=True)
sns.scatterplot(data=gdf_metrics,x='NNSE_A',y='lnNNSE_A',size='PBIAS_A',hue='Climate',style='Climate',palette=['r','b','gray'],ax=ax[0,],edgecolor='k',lw=0.3,legend=False,alpha=0.4,sizes=(50,250))
sns.scatterplot(data=gdf_metrics,x='NNSE_S',y='lnNNSE_S',size='PBIAS_S',hue='Climate',style='Climate',palette=['r','b','gray'],ax=ax[1],edgecolor='k',lw=0.3,legend=False,alpha=0.4,sizes=(50,250))
sns.scatterplot(data=gdf_metrics,x='NNSE_W',y='lnNNSE_W',size='PBIAS_W',hue='Climate',style='Climate',palette=['r','b','gray'],ax=ax[2],edgecolor='k',lw=0.3,legend=False,alpha=0.4,sizes=(50,250))
ax[0].set_xlim(0,1)
ax[0].set_ylim(0,1)
ax[0].set_aspect('equal')
ax[1].set_xlim(0,1)
ax[1].set_ylim(0,1)
ax[1].set_aspect('equal')
ax[2].set_xlim(0,1)
ax[2].set_ylim(0,1)
ax[2].set_aspect('equal')
sns.regplot(data=gdf_metrics[gdf_metrics['Climate']=='M'],x='NNSE_A',y='lnNNSE_A',scatter=False,ax=ax[0],color='r')
sns.regplot(data=gdf_metrics[gdf_metrics['Climate']=='W'],x='NNSE_A',y='lnNNSE_A',scatter=False,ax=ax[0],color='b')
sns.regplot(data=gdf_metrics[gdf_metrics['Climate']=='O'],x='NNSE_A',y='lnNNSE_A',scatter=False,ax=ax[0],color='k')


# for i,climate in enumerate(gdf_metrics['Climate'].unique()):
#     for j,season in enumerate(['A','S','W']):

#         df = gdf_metrics[(gdf_metrics['Climate']==climate)]
#         n_sites = len(df)
        # sns.scatterplot(data=df,x='NNSE_{}'.format(season),y='lnNNSE_{}'.format(season),hue='PBIAS_{}'.format(season),hue_norm=cmap_prop['PBIAS']['norm'],
        #                 ax=ax[i,j],edgecolor='k',lw=0.3,legend=False,palette=cmap_prop['PBIAS']['cmap'],s=50)
        # ax[i,j].set_title('Climate: {} | Season: {}'.format(climate,season))
        # ax[i,j].grid(True,color='gray',linestyle='--',linewidth=0.5)
        # ax[i,j].set_axisbelow(True)
        # ax[i,j].set_xlim(0,1)
        # ax[i,j].set_ylim(0,1)
        # ax[i,j].set_aspect('equal')
        # ax[i,j].xaxis.set_major_locator(ticker.MultipleLocator(0.1))
        # ax[i,j].xaxis.set_major_formatter(ticker.ScalarFormatter())
        # ax[i,j].yaxis.set_major_locator(ticker.MultipleLocator(0.1))
        # ax[i,j].yaxis.set_major_formatter(ticker.ScalarFormatter())
        # ax[i,j].axvline(0.5,color='k',linestyle='--',linewidth=1)
        # ax[i,j].axhline(0.5,color='k',linestyle='--',linewidth=1)
        # ax[i,j].set_xlabel('NNSE',fontsize=12)
        # ax[i,j].set_ylabel('LNNSE',fontsize=12)
        # ax[i,j].text(0.05,0.9,'N = {}'.format(n_sites),transform=ax[i,j].transAxes,fontsize=12)
# plt.tight_layout()
# plt.subplots_adjust(hspace = 0.1)
# plt.subplots_adjust(wspace = 0.1)
# cbar_ax = fig.add_axes([1.01, 0.2, 0.03, 0.6])
# cb = mpl.colorbar.ColorbarBase(cmap=cmap_prop['PBIAS']['cmap'],
#                     norm=cmap_prop['PBIAS']['norm'],
#                     ticks=cmap_prop['PBIAS']['bounds'],
#                     ax=cbar_ax)
# cb.ax.tick_params(labelsize=12) 
# cb.ax.set_title('PBIAS (%)',y=1.02,fontsize=12)

# fig.savefig(os.path.join(savedir,'TS_scatter_metrics.png'),dpi=300,bbox_inches='tight')
# fig.savefig(os.path.join(savedir,'TS_scatter_metrics.pdf'),dpi=300,bbox_inches='tight')

# # Percentage Scatter Plot
# df_percentage = pd.DataFrame(columns=['Climate','Season','NNSE','LNNSE','PBIAS'])
# for climate in gdf_metrics['Climate'].unique():
#     for season in ['A','S','W']:

#         # NNSE
#         NNSE = gdf_metrics[(gdf_metrics['Climate']==climate)]['NNSE_{}'.format(season)]
#         NNSE_50p = ((NNSE[NNSE>0.5]).count()/NNSE.count())*100

#         # LNNSE
#         LNNSE = gdf_metrics[(gdf_metrics['Climate']==climate)]['lnNNSE_{}'.format(season)]
#         LNNSE_50p = ((LNNSE[LNNSE>0.5]).count()/LNNSE.count())*100

#         # PBIAS
#         PBIAS = gdf_metrics[(gdf_metrics['Climate']==climate)]['PBIAS_{}'.format(season)]
#         PBIAS_50p = ((PBIAS[PBIAS.abs()<50]).count()/PBIAS.count())*100

#         df_percentage.loc[len(df_percentage)] = [climate,season,NNSE_50p,LNNSE_50p,PBIAS_50p]

#         # df_percentage.loc[climate,season] = [NNSE.mean(),NNSE.std()]
#         # df = gdf_metrics[(gdf_metrics['Climate']==climate) & (gdf_metrics['Season']==season)]
#         # df_percentage.loc[climate,season] = [df['NNSE'].mean(),df['LNNSE'].mean(),df['PBIAS'].mean()]

# fig,ax = plt.subplots(figsize=(3,3))
# sns.scatterplot(data=df_percentage,x='NNSE',y='LNNSE',hue='Climate',style='Season',size='PBIAS',ax=ax,edgecolor='k',lw=0.5,palette=['r','b','gray'],sizes=(50,250))
# ax.set_xlabel('Sites with NNSE > 0.5 (%)',fontsize=12)
# ax.set_ylabel('Sites with LNNSE > 0.5 (%)',fontsize=12)
# ax.legend(loc='upper left',ncols=1)
# ax.grid(True,color='gray',linestyle='--',linewidth=0.5)
# ax.set_axisbelow(True)
# ax.set_xlim(0,80)
# ax.set_ylim(0,80)
# ax.set_aspect('equal')
# sns.move_legend(ax, "upper left", bbox_to_anchor=(1.1, 1))
# ax.xaxis.set_major_locator(ticker.MultipleLocator(10))
# ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
# ax.yaxis.set_major_locator(ticker.MultipleLocator(10))
# ax.yaxis.set_major_formatter(ticker.ScalarFormatter())

# fig.savefig(os.path.join(savedir,'TS_scatter_metrics_summary.png'),dpi=300,bbox_inches='tight')
# fig.savefig(os.path.join(savedir,'TS_scatter_metrics_summary.pdf'),dpi=300,bbox_inches='tight')
