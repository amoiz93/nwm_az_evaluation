import os
import sys
sys.path.append('../src')
import glob
import xarray as xr
import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import box
import pyogrio
import fiona
import rasterio
import rioxarray
import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import cartopy.io.img_tiles as cimgt
import cartopy.feature as cfeature
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches
import colormaps as cmaps
import PIL
import nwm
import param_nwm3
import misc



import seaborn as sns
#sns.set_style("ticks")
sns.set_style("ticks",{'axes.grid' : True})

savedir = os.path.join(param_nwm3.out_dir,'figures',os.path.basename(__file__).split('.')[0])
misc.makedir(savedir)

gdf_site_info = gpd.read_parquet('../out/05_data_availability/usgs_q_info_flagged_utm12n_nad83.parquet.gzip')
gdf_site_info = gdf_site_info[gdf_site_info['yuma']==0] # Exclude YUMA stations
# gdf_site_info = gdf_site_info[(gdf_site_info['hcdn2009']==1)]

#gdf_site_info = gdf_site_info[gdf_site_info['site_tp_cd'].isin(['ST','LK'])] 


# gdf_site_info_daily = gdf_site_info[gdf_site_info['daily']==1]
# gdf_site_info_hourly = gdf_site_info[gdf_site_info['inst']==1]

# gdf_site_info_daily['nid_maxstorage'] = gdf_site_info_daily['nid_maxstorage']/1E06
# gdf_site_info_hourly['nid_maxstorage'] = gdf_site_info_hourly['nid_maxstorage']/1E06

# fig,ax = plt.subplots(figsize=(4,4))
# sns.ecdfplot(data=gdf_site_info_daily,x='nid_maxstorage',ax=ax,color='k',label='NID Storage',stat='count')
# sns.ecdfplot(data=gdf_site_info_hourly,x='nid_maxstorage',ax=ax,color='r',label='NID Storage',stat='count')
# plt.xscale('log')

# Daily
gdf_site_info_daily = gdf_site_info[gdf_site_info['daily']==1]
maxstorage_value = gdf_site_info_daily['nid_maxstorage'].max()
maxstorage_hist = np.histogram((gdf_site_info_daily['nid_maxstorage']).values,bins=range(0,int(maxstorage_value),10000))
maxstorage_hist = pd.DataFrame(index=maxstorage_hist[1][0:-1],data=maxstorage_hist[0])
sites_maxstorage_cdf=maxstorage_hist.cumsum()
sites_maxstorage_cdf_index = sites_maxstorage_cdf.index.tolist()
sites_maxstorage_cdf_index = [0.001 if x==0 else x for x in sites_maxstorage_cdf_index]
sites_maxstorage_cdf.index = sites_maxstorage_cdf_index
sites_maxstorage_cdf_daily = sites_maxstorage_cdf
sites_maxstorage_cdf_daily.columns = ['Daily']
sites_maxstorage_cdf_daily.index = sites_maxstorage_cdf_daily.index/1E06 # MCM

# Inst
gdf_site_info_inst = gdf_site_info[gdf_site_info['inst']==1]
maxstorage_value = gdf_site_info_inst['nid_maxstorage'].max()
maxstorage_hist = np.histogram((gdf_site_info_inst['nid_maxstorage']).values,bins=range(0,int(maxstorage_value),10000))
maxstorage_hist = pd.DataFrame(index=maxstorage_hist[1][0:-1],data=maxstorage_hist[0])
sites_maxstorage_cdf=maxstorage_hist.cumsum()
sites_maxstorage_cdf_index = sites_maxstorage_cdf.index.tolist()
sites_maxstorage_cdf_index = [0.001 if x==0 else x for x in sites_maxstorage_cdf_index]
sites_maxstorage_cdf.index = sites_maxstorage_cdf_index
sites_maxstorage_cdf_inst = sites_maxstorage_cdf
sites_maxstorage_cdf_inst.columns = ['Hourly']
sites_maxstorage_cdf_inst.index = sites_maxstorage_cdf_inst.index/1E06 # MCM


figsize = (5,5)
lw=1.0
title = 'Total reservoir storage \nupstream of USGS gages'
title_fontsize = 14
legend_title = 'Frequency'
legend_title_fontsize = 12
axis_xlabel = 'Reservoir storage (Mm$^3$)'
axis_ylabel = 'Number of gages'
axis_ticklabel_fontsize = 12
axis_label_fontsize = 14

fig,ax = plt.subplots(figsize=figsize)
sites_maxstorage_cdf_daily.plot(ax=ax,legend=True,style='-k',lw=lw)
sites_maxstorage_cdf_inst.plot(ax=ax,legend=True,style='-r',lw=lw)
#ax.invert_xaxis()
plt.xscale("log")
ax.set_xlabel(axis_xlabel,fontsize=axis_label_fontsize)
ax.set_ylabel(axis_ylabel,fontsize=axis_label_fontsize)
ax.tick_params(labelsize=axis_ticklabel_fontsize)
ax.legend(frameon=True,prop={'size':12})
ax.set_ylim(bottom=100)
ax.set_xlim(left=1E-02,right=1E05)
ax.axvline(x=10**8/1E06,lw=lw,ls='--',color='k')
#ax.set_title(title,fontsize=title_fontsize)
plt.tight_layout()
fig.savefig(os.path.join(savedir,'nid_maxstorage_hist.pdf'),dpi=300,bbox_inches='tight')



# sites_maxstorage_cdf=test.cumsum()



# df_yuma_stations = pd.read_csv('../inp/yuma_excluded.csv')
# gdf_site_info = pyogrio.read_dataframe('../out/02_check_flags_USGS_Q_data/usgs_q_info_flagged.gpkg')
# gdf_site_info = gdf_site_info[gdf_site_info['yuma']==0] # Exclude YUMA stations
# gdf_site_info = gdf_site_info[~gdf_site_info['site_no'].isin(df_yuma_stations['site_no'])]
# gdf_site_info = gdf_site_info[gdf_site_info['nid_maxstorage'].notnull()]

# # Daily
# gdf_site_info_daily = gdf_site_info[gdf_site_info['daily']==1]

# max_storage_value = gdf_site_info_daily['nid_maxstorage'].max()

# test = np.histogram((gdf_site_info_daily['nid_maxstorage']).values,bins=range(0,int(max_storage_value),10000))
# test=pd.DataFrame(index=test[1][0:-1],data=test[0])
# sites_maxstorage_cdf=test.cumsum()
# #sites_maxstorage_cdf.index = sites_maxstorage_cdf.index/1E06 # MCM
# sites_maxstorage_cdf_index = sites_maxstorage_cdf.index.tolist()
# sites_maxstorage_cdf_index = [0.001 if x==0 else x for x in sites_maxstorage_cdf_index]
# sites_maxstorage_cdf.index = sites_maxstorage_cdf_index
# sites_maxstorage_cdf_daily = sites_maxstorage_cdf
# sites_maxstorage_cdf_daily.columns = ['Daily']

# # Inst
# gdf_site_info_inst = gdf_site_info[gdf_site_info['inst']==1]

# max_storage_value = gdf_site_info_inst['nid_maxstorage'].max()
# test = np.histogram((gdf_site_info_inst['nid_maxstorage']).values,bins=range(0,int(max_storage_value),10000))
# test=pd.DataFrame(index=test[1][0:-1],data=test[0])
# sites_maxstorage_cdf=test.cumsum()
# #sites_maxstorage_cdf.index = sites_maxstorage_cdf.index/1E06 # MCM
# sites_maxstorage_cdf_index = sites_maxstorage_cdf.index.tolist()
# sites_maxstorage_cdf_index = [0.001 if x==0 else x for x in sites_maxstorage_cdf_index]
# sites_maxstorage_cdf.index = sites_maxstorage_cdf_index
# sites_maxstorage_cdf_inst = sites_maxstorage_cdf
# sites_maxstorage_cdf_inst.columns = ['Inst']

# fig,ax = plt.subplots()
# sites_maxstorage_cdf_daily.plot(ax=ax,legend=True,style='-k')
# sites_maxstorage_cdf_inst.plot(ax=ax,legend=True,style='-r')
# ax.invert_xaxis()
# plt.xscale("log")
# ax.set_xlabel('NID total reservoir storage upstream of a gage (m$^3$)')
# ax.set_ylabel('Number of gages')
# ax.set_ylim(bottom=0)
# plt.tight_layout()
# fig.savefig(os.path.join(savedir,'nid_maxstorage_hist.png'),dpi=300,bbox_inches='tight')

