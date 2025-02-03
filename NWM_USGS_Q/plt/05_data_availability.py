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
import seaborn as sns
import nwm
import param_nwm3
import misc
import usgs

import seaborn as sns
#sns.set_style("ticks")
sns.set_style("ticks",{'axes.grid' : True})


savedir = os.path.join(param_nwm3.out_dir,'figures',os.path.basename(__file__).split('.')[0])
misc.makedir(savedir)

start_year = 2004
end_year = 2023
years_range = range(start_year,end_year+1)
nid_store_thres = 10**8
p_thresholds = [70,80,90,95,100]

gdf_site_info = gpd.read_parquet('../out/05_data_availability/usgs_q_info_flagged_utm12n_nad83.parquet.gzip')
gdf_site_info = gdf_site_info[gdf_site_info['yuma']==0] # Exclude YUMA stations
gdf_site_info = gdf_site_info[(gdf_site_info['nid_maxstorage']<nid_store_thres)|
                              (gdf_site_info['hcdn2009']==1)]


gdf_site_info1 = usgs.list_of_gages_with_pavail_start_end_year(gdf_site_info,'daily',start_year,end_year,95)
gdf_site_info2 = usgs.list_of_gages_with_pavail_nyear(gdf_site_info,'daily',95,20)
gdf_site_info3 = usgs.list_of_gages_with_pavail(gdf_site_info,'daily',start_year,end_year,20,95)


df_daily_avail = usgs.calc_ngage_availability_by_threshold(gdf_site_info,'daily',p_thresholds)
df_inst_avail = usgs.calc_ngage_availability_by_threshold(gdf_site_info,'hourly',p_thresholds)

# test = usgs.filter_stations(gdf_site_info,'daily',[2003,2022],nid_store_thres,95)


lw=1.0
axis_label_fontsize = 14
axis_ticklabel_fontsize = 12
title_fontsize = 14
legend_title_fontsize = 12
fig, ax = plt.subplots(figsize=(5,5))

# Format Columns
df_daily_avail.columns = [('$\it{p}$'+'='+p.split('=')[-1].split('%')[0]+'%') for p in df_daily_avail.columns]
df_inst_avail.columns = [('$\it{p}$'+'='+p.split('=')[-1].split('%')[0]+'%') for p in df_inst_avail.columns]

df_daily_avail.plot(ax=ax,ls='-',color=sns.color_palette('Grays'),lw=lw)
df_inst_avail.plot(ax=ax,ls='--',legend=False,color=sns.color_palette('Reds'),lw=lw)
ax.legend(title='Daily'+' ' * 17+'Hourly',frameon=True,ncols=2,title_fontsize=12,prop={'size':10})
ax.set_ylabel('Number of gages with $\it{p} (\%)\ $data',fontsize=axis_label_fontsize)
ax.set_xlabel('Number of years',fontsize=axis_label_fontsize)
ax.set_xlim(left=0)
ax.set_ylim(bottom=0)
ax.tick_params(labelsize=axis_ticklabel_fontsize)
ax.axvline(x=20,lw=lw,ls='--',color='k')
#ax.set_title('Data Availability at USGS Stations\n[Total upstream reservoir area $\leq$ 100 Mm$^3$]',fontsize=title_fontsize)
plt.tight_layout()
fig.savefig(os.path.join(savedir,'data_availability.pdf'),dpi=300,bbox_inches='tight')


