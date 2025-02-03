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
from matplotlib.lines import Line2D
import seaborn as sns

import custom_basemaps as cbm

import param_nwm3
import misc


def positive_log(x):
    if x>0:
        return np.log(x)
    else:
        return x

def signed_log(x):
    if x>0:
        return np.log(x)
    elif x<0:
        return -np.log(-x)
    elif x==0:
        return 0
    else:
        return x

savedir = os.path.join(param_nwm3.out_dir,'figures',os.path.basename(__file__).split('.')[0])
misc.makedir(savedir)

metrics_dir = '/Users/amoiz2/Servers/dropbox_asu/Personal/Work/20230807_ASU_SSEBE_Mascaro/Workspace/projects/20240311_USGS_NWM_Q_Validation_AZ/out/06A_add_attributes/parquet'
start_year = 2003
end_year = 2022
frequency = 'hourly'

pq_metrics = os.path.join(metrics_dir,'{}_{}_{}_utm12n_nad83.parquet.gzip'.format(frequency,str(start_year),str(end_year)))
gdf_metrics = gpd.read_parquet(pq_metrics)

cmap_prop = {
    'NNSE':{'bounds':np.array([0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]),
            'norm':colors.BoundaryNorm(boundaries=np.array([0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]), ncolors=256),
            'cmap':'turbo_r'},

    'NSE':{'bounds':np.array([0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]),
            'norm':colors.BoundaryNorm(boundaries=np.array([0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]), ncolors=256),
            'cmap':'turbo_r'},
            
    'PBIAS':{'bounds':np.array(np.linspace(-100,100,21)),
            'norm':colors.BoundaryNorm(boundaries=np.linspace(-100,100,21), ncolors=256),
            'cmap':'RdYlBu'},

    'PBIAS_LF':{'bounds':np.array(np.linspace(-200,200,41)),
            'norm':colors.BoundaryNorm(boundaries=np.linspace(-200,200,41), ncolors=256),
            'cmap':'RdYlBu'},

    'BIAS':{'bounds':np.array([-50,-40,-30,-20,-10,0,10,20,30,40,50]),
            'norm':colors.BoundaryNorm(boundaries=np.array([-50,-40,-30,-20,-10,0,10,20,30,40,50]), ncolors=256),
            'cmap':'RdYlBu'},
}

climate_group_label_properties = {'label_dict':{'M':'MON','W':'WIN','O':'OTH'},
                                'label_fontsize':12,
                                'label_color':{'M':'r','W':'b','O':'gray'},
                                'label_weight':'bold',}
season_group_label_properties = {'label_dict':{'A':'WY','S':'S','W':'W'},}

cmap = mpl.colormaps.get_cmap(cmap_prop['PBIAS']['cmap'])
cmap.set_over('k')

# Metrics Scatter Plot
fig,ax = plt.subplots(3,3,figsize=(10,11),sharex=True,sharey=True)

for i,climate in enumerate(gdf_metrics['Climate'].unique()):
    for j,season in enumerate(['A','S','W']):

        df = gdf_metrics[(gdf_metrics['Climate']==climate)].copy()
        # df.loc[,'lw'] = 1
        # df.loc[df['PBIAS_{}'.format(season)].abs()>50,'lw'] = 0.1
        # df['lw'] = df['PBIAS_{}'.format(season)].abs()
        
        
        pbias_threshold = 200
        # df = df[(df['PBIAS_HF_{}'.format(season)].abs()<=pbias_threshold) & (df['PBIAS_LF_{}'.format(season)].abs()<=pbias_threshold)]
        # print(df['PBIAS_LF_{}'.format(season)])

        # test1 = df.copy()
        # df['PBIAS_HF_{}'.format(season)] = df['PBIAS_HF_{}'.format(season)].apply(signed_log)
        # df['PBIAS_LF_{}'.format(season)] = df['PBIAS_LF_{}'.format(season)].apply(signed_log)
        

        pbias_lf = df['PBIAS_LF_{}'.format(season)]
        
        n_sites_inf = np.isinf(pbias_lf).sum()
        n_sites = len(pbias_lf[pbias_lf.notnull()]) - n_sites_inf
        
        

        # base = df[['PBIAS_HF_{}'.format(season),'PBIAS_LF_{}'.format(season)]]
        # test = df[['PBIAS_HF_{}'.format(season),'PBIAS_LF_{}'.format(season)]].applymap(positive_log)
        # test1 = df[['PBIAS_HF_{}'.format(season),'PBIAS_LF_{}'.format(season)]].applymap(signed_log)

        # # NNSE
        # sns.scatterplot(data=df,x='PBIAS_HF_{}'.format(season),y='PBIAS_LF_{}'.format(season),hue='NNSE_{}'.format(season),hue_norm=cmap_prop['NNSE']['norm'],
        #                 ax=ax[i,j],edgecolor='k',lw=0.2,legend=False,palette=cmap,s=40,zorder=2,alpha=1)

        # # LNNSE
        # sns.scatterplot(data=df,x='PBIAS_HF_{}'.format(season),y='PBIAS_LF_{}'.format(season),hue='lnNNSE_{}'.format(season),hue_norm=cmap_prop['NNSE']['norm'],
        #                 ax=ax[i,j],edgecolor='k',lw=0.2,legend=False,palette=cmap,s=40,zorder=2,alpha=1)

        # # PBIAS
        # sns.scatterplot(data=df,x='PBIAS_HF_{}'.format(season),y='PBIAS_LF_{}'.format(season),hue='PBIAS_{}'.format(season),hue_norm=cmap_prop['PBIAS']['norm'],
        #                ax=ax[i,j],edgecolor='k',lw=0.2,legend=False,palette=cmap,s=40,zorder=2,alpha=1)   

        # (1) NNSE > 0.5 & LNNSE > 0.5
        # (2) NNSE > 0.5 & LNNSE <= 0.5
        # (3) NNSE <= 0.5 & LNNSE > 0.5
        # (4) NNSE <= 0.5 & LNNSE <= 0.5

        # Color Hue: PBIAS
        # sns.scatterplot(data=df[(df['NNSE_{}'.format(season)]>0.5)&(df['lnNNSE_{}'.format(season)]>0.5)],x='PBIAS_HF_{}'.format(season),y='PBIAS_LF_{}'.format(season),hue='PBIAS_{}'.format(season),hue_norm=cmap_prop['PBIAS']['norm'],
        #                 ax=ax[i,j],edgecolor='k',lw=0.2,legend=False,palette=cmap,s=40,zorder=5,alpha=1,marker='o')
        # sns.scatterplot(data=df[(df['NNSE_{}'.format(season)]>0.5)&(df['lnNNSE_{}'.format(season)]<=0.5)],x='PBIAS_HF_{}'.format(season),y='PBIAS_LF_{}'.format(season),hue='PBIAS_{}'.format(season),hue_norm=cmap_prop['PBIAS']['norm'],
        #                 ax=ax[i,j],edgecolor='k',lw=0.2,legend=False,palette=cmap,s=40,zorder=4,alpha=1,marker='>')
        # sns.scatterplot(data=df[(df['NNSE_{}'.format(season)]<=0.5)&(df['lnNNSE_{}'.format(season)]>0.5)],x='PBIAS_HF_{}'.format(season),y='PBIAS_LF_{}'.format(season),hue='PBIAS_{}'.format(season),hue_norm=cmap_prop['PBIAS']['norm'],
        #                 ax=ax[i,j],edgecolor='k',lw=0.2,legend=False,palette=cmap,s=40,zorder=3,alpha=1,marker='<')
        # sns.scatterplot(data=df[(df['NNSE_{}'.format(season)]<=0.5)&(df['lnNNSE_{}'.format(season)]<=0.5)],x='PBIAS_HF_{}'.format(season),y='PBIAS_LF_{}'.format(season),hue='PBIAS_{}'.format(season),hue_norm=cmap_prop['PBIAS']['norm'],
        #                 ax=ax[i,j],edgecolor='k',lw=0.2,legend=False,palette=cmap,s=40,zorder=2,alpha=1,marker='s')

        # No Hue: Colors are categorized by NNSE and LNNSE condition
        sns.scatterplot(data=df[(df['NNSE_{}'.format(season)]>0.5)&(df['lnNNSE_{}'.format(season)]>0.5)],x='PBIAS_HF_{}'.format(season),y='PBIAS_LF_{}'.format(season),color='b',
                        ax=ax[i,j],edgecolor='k',lw=0.2,legend=False,palette=cmap,s=40,zorder=5,alpha=1,marker='o')
        sns.scatterplot(data=df[(df['NNSE_{}'.format(season)]>0.5)&(df['lnNNSE_{}'.format(season)]<=0.5)],x='PBIAS_HF_{}'.format(season),y='PBIAS_LF_{}'.format(season),color='g',
                        ax=ax[i,j],edgecolor='k',lw=0.2,legend=False,palette=cmap,s=50,zorder=4,alpha=1,marker='>')
        sns.scatterplot(data=df[(df['NNSE_{}'.format(season)]<=0.5)&(df['lnNNSE_{}'.format(season)]>0.5)],x='PBIAS_HF_{}'.format(season),y='PBIAS_LF_{}'.format(season),color='yellow',
                        ax=ax[i,j],edgecolor='k',lw=0.2,legend=False,palette=cmap,s=50,zorder=3,alpha=1,marker='<')
        sns.scatterplot(data=df[(df['NNSE_{}'.format(season)]<=0.5)&(df['lnNNSE_{}'.format(season)]<=0.5)],x='PBIAS_HF_{}'.format(season),y='PBIAS_LF_{}'.format(season),color='r',
                        ax=ax[i,j],edgecolor='k',lw=0.2,legend=False,palette=cmap,s=30,zorder=2,alpha=1,marker='s')

        n_sites_quad1 = len(df[(df['PBIAS_LF_{}'.format(season)]>0)&(df['PBIAS_HF_{}'.format(season)]<0)])
        n_sites_quad2 = len(df[(df['PBIAS_LF_{}'.format(season)]>0)&(df['PBIAS_HF_{}'.format(season)]>0)])
        n_sites_quad3 = len(df[(df['PBIAS_LF_{}'.format(season)]<0)&(df['PBIAS_HF_{}'.format(season)]>0)])
        n_sites_quad4 = len(df[(df['PBIAS_LF_{}'.format(season)]<0)&(df['PBIAS_HF_{}'.format(season)]<0)])
        

        # # Color Hue: NNSE
        # sns.scatterplot(data=df,x='PBIAS_HF_{}'.format(season),y='PBIAS_LF_{}'.format(season),hue='NNSE_{}'.format(season),hue_norm=cmap_prop['NNSE']['norm'],
        #                 ax=ax[i,j],edgecolor='k',lw=0.2,legend=False,palette=cmap,s=40,zorder=5,alpha=1,marker='o')

        if i==0:
            ax[i,j].set_title('{}'.format(season_group_label_properties['label_dict'][season]),fontsize=20,weight='bold')
        ax[i,j].text(0.04,0.90,climate_group_label_properties['label_dict'][climate],transform=ax[i,j].transAxes,fontsize=20,fontweight=climate_group_label_properties['label_weight'],color=climate_group_label_properties['label_color'][climate])
        # ax[i,j].text(0.80,0.99,'$n$ = '+str(n_sites)+'\n$n_{inf}$ = '+str(n_sites_inf),transform=ax[i,j].transAxes,fontsize=16,ha='right',va='top')
        ax[i,j].text(0.80,0.99,'$n$ = '+str(n_sites),transform=ax[i,j].transAxes,fontsize=16,ha='right',va='top')
        ax[i,j].grid(True,color='gray',linestyle='--',linewidth=0.5)
        ax[i,j].set_axisbelow(True)
        # ax[i,j].set_xlim(left=-100)
        # ax[i,j].set_ylim(bottom=-100)
        ax[i,j].plot([-130, 1E5], [-130, 1E5], color='k', linestyle='--', linewidth=1)
        ax[i,j].set_xlim(-130,1E5)
        ax[i,j].set_ylim(-130,1E5)
        ax[i,j].set_aspect('equal')
        # ax[i,j].xaxis.set_major_locator(ticker.MultipleLocator(2))
        # ax[i,j].xaxis.set_major_formatter(ticker.ScalarFormatter())
        # ax[i,j].yaxis.set_major_locator(ticker.MultipleLocator(2))
        # ax[i,j].yaxis.set_major_formatter(ticker.ScalarFormatter())
        ax[i,j].tick_params(axis='both', which='major', labelsize=14)
        ax[i,j].tick_params(axis='x', which='major', labelsize=14,rotation=0)
        ax[i,j].axvline(0,color='k',linestyle='--',linewidth=1,zorder=1)
        ax[i,j].axhline(0,color='k',linestyle='--',linewidth=1,zorder=1)
        ax[i,j].set_xlabel('PBIAS$_\mathrm{HF}$ (%)',fontsize=22)
        ax[i,j].set_ylabel('PBIAS$_\mathrm{LF}$ (%)',fontsize=22)
        ax[i,j].set_yscale('symlog',linthresh=100)
        ax[i,j].set_xscale('symlog',linthresh=100)
        
        
        # # Add Quadrant Labels
        # ax[i,j].text(-0.05,0.05,ha='right',va='bottom',fontsize=16,s=str(n_sites_quad1))
        # ax[i,j].text(0.05,0.05,ha='left',va='bottom',fontsize=16,s=str(n_sites_quad2))
        # ax[i,j].text(0.05,-0.05,ha='left',va='top',fontsize=16,s=str(n_sites_quad3))
        # ax[i,j].text(-0.05,-0.05,ha='right',va='top',fontsize=16,s=str(n_sites_quad4))

# Create custom legend handles and labels
legend_elements = [Line2D([0], [0], marker='o', color='None',markeredgecolor='k', markerfacecolor='b', markersize=14),
                   Line2D([0], [0], marker='>', color='None',markeredgecolor='k', markerfacecolor='g', markersize=14),
                   Line2D([0], [0], marker='<', color='None',markeredgecolor='k', markerfacecolor='yellow', markersize=14),
                   Line2D([0], [0], marker='s', color='None',markeredgecolor='k', markerfacecolor='r', markersize=14)]
legend_labels = ['NNSE > 0.5, LNNSE > 0.5','NNSE > 0.5, LNNSE <= 0.5','NNSE <= 0.5, LNNSE > 0.5','NNSE <= 0.5, LNNSE <= 0.5']
ax[2,1].legend(handles=legend_elements, labels=legend_labels, loc='upper center',ncols=2,prop={'size': 18},bbox_to_anchor=(0.5, -0.3),frameon=False)

plt.tight_layout()
plt.subplots_adjust(hspace = 0.05)
plt.subplots_adjust(wspace = 0.15)
plt.subplots_adjust(bottom=0.2)
# cbar_ax = fig.add_axes([1.02, 0.25, 0.03, 0.6])

# cb = mpl.colorbar.ColorbarBase(cmap=cmap,
#                     norm=cmap_prop['PBIAS']['norm'],
#                     ticks=cmap_prop['PBIAS']['bounds'],
#                     ax=cbar_ax,
#                     extend='max')
# cb.ax.tick_params(labelsize=16) 
# cb.ax.set_title('PBIAS (%)',x=1.04,y=1.06,fontsize=22)


fig.savefig(os.path.join(savedir,'TS_scatter_metrics_{}_all_conditions.png'.format(frequency)),dpi=300,bbox_inches='tight')
fig.savefig(os.path.join(savedir,'TS_scatter_metrics_{}_all_conditions.pdf'.format(frequency)),dpi=300,bbox_inches='tight')
fig.savefig(os.path.join(savedir,'TS_scatter_metrics_{}_all_conditions.svg'.format(frequency)),dpi=300,bbox_inches='tight')
