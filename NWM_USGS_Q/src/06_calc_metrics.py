import os
import sys
sys.path.append('../src')

import xarray as xr
import numpy as np
import pandas as pd

import geopandas as gpd
import pyogrio
import fiona
import pyproj

import dataretrieval.nwis as nwis

import usgs
import nid
import nwm
import param_nwm3
import misc
import eval_metrics


import matplotlib.pyplot as plt
import matplotlib


matplotlib.use('Agg')

# import warnings
# warnings.filterwarnings("ignore")

# Making output directories
savedir = os.path.join(param_nwm3.out_dir,os.path.basename(__file__).split('.')[0])
savedir_gpkg = os.path.join(savedir,'gpkg')
savedir_shp = os.path.join(savedir,'shp')
savedir_pkl = os.path.join(savedir,'pkl')
savedir_csv = os.path.join(savedir,'csv')
savedir_parquet = os.path.join(savedir,'parquet')
misc.makedir(savedir_gpkg)
misc.makedir(savedir_shp)
misc.makedir(savedir_pkl)
misc.makedir(savedir_csv)
misc.makedir(savedir_parquet)

savedir_fig = os.path.join(param_nwm3.out_dir,'figures',os.path.basename(__file__).split('.')[0])
misc.makedir(savedir_fig)



# Read Inputs
usgs_data_dir = '../out/03_reformat_USGS_Q_data/usgs/parquet/'
nwm_data_dir = '../out/04_nwm3_data_download/nwm3/parquet/'
gdf_site_info = gpd.read_parquet(os.path.join('../out/05_data_availability','usgs_q_info_flagged_utm12n_nad83.parquet.gzip'))
# gdf_site_info = gdf_site_info.to_crs(param_nwm3.crs_utm12n_nad83)
gdf_watersheds = pyogrio.read_dataframe(os.path.join('../../../data/az_nwm/const/basins/az_watersheds_simplified_utm12n_nad83.shp')).to_crs(gdf_site_info.crs)
gdf_nwm21_calib_basins = pyogrio.read_dataframe('../inp/NWMv2.1_calibration_basins/Final_V2.1_Calibration_Basins_CONUS_GreatLake_Size_limited.shp').to_crs(gdf_site_info.crs)


nid_store_thres = 10**8
p_thres_dict = {'hourly':80,
                'daily':95,
                'monthly':95,
                'yearly':95}

plot_figures = False # Flag to plot timeseries #TODO: remove this
nyears=20
frequencies = [
               'hourly',
               'daily',
            #    'monthly',
            #    'yearly' # Do not use this for yearly
              ]

periods = [
        #    [1980,2022],
        #    [1980,2001],
        #    [2002,2022],
        #    [2002,2015],
        #    [2016,2022],
        #    [2013,2022],
           [2003,2022]
           ]

seasons = {
           'all':{'months':[1,2,3,4,5,6,7,8,9,10,11,12],'symbol':'A'},   # Jan - Dec
           'winter':{'months':[11,12,1,2,3],'symbol':'W'}, # Nov - Mar
           'summer':{'months':[7,8,9],'symbol':'S'} # Jul - Sep
          }

# exceedances = [5,20,70,95,100]
# exceedances = [10,25,50,75,90,100]
exceedances = [10,50,90,100]

groups = []

for i in range(len(frequencies)):
    for j in range(len(periods)):
        groups.append((frequencies[i],periods[j]))

for group in groups:
    frequency = group[0]
    period = group[1]

    # start_time = '{}-01-01 00:00:00'.format(str(period[0]))
    # end_time = '{}-12-31 23:00:00'.format(str(period[1]))

    start_wy = period[0]
    end_wy = period[1]

    gdf_site_info_group_metrics = gdf_site_info.copy()

    if frequency == 'hourly':
        gdf_site_info_group_metrics = gdf_site_info_group_metrics[gdf_site_info_group_metrics['inst']==1]
    else:
        gdf_site_info_group_metrics = gdf_site_info[gdf_site_info['daily']==1]
    gdf_site_info_group_metrics = gdf_site_info_group_metrics[(gdf_site_info_group_metrics['nwm3p_comid'].notnull()) |
                                            (gdf_site_info_group_metrics['nwm3g_comid'].notnull())]
    #gdf_site_info_group_metrics = gdf_site_info_group_metrics[gdf_site_info_group_metrics['site_no'] == '09484500']

    # gdf_site_info_group_metrics,dict_site_info_avail_years = usgs.filter_stations(gdf_site_info_group_metrics,frequency,[int(start_wy),int(end_wy)],nyears,nid_store_thres,p_thres_dict[frequency])
    if frequency in ['daily','monthly','yearly']:
        gdf_site_info_group_metrics_A,dict_site_info_avail_years_A = usgs.filter_stations(gdf_site_info_group_metrics,frequency,[int(start_wy),int(end_wy)],nyears,nid_store_thres,p_thres_dict[frequency])
        
        gdf_site_info_group_metrics_B = gdf_site_info.copy()
        gdf_site_info_group_metrics_B = gdf_site_info_group_metrics_B[(gdf_site_info_group_metrics_B['nwm3p_comid'].notnull()) |
                                        (gdf_site_info_group_metrics_B['nwm3g_comid'].notnull())]
        gdf_site_info_group_metrics_B = gdf_site_info_group_metrics_B[gdf_site_info_group_metrics_B['inst']==1]
        gdf_site_info_group_metrics_B,dict_site_info_avail_years_B = usgs.filter_stations(gdf_site_info_group_metrics_B,'hourly',[int(start_wy),int(end_wy)],nyears,nid_store_thres,p_thres_dict['hourly'])
        usgs_sites_B = gdf_site_info_group_metrics_B[~gdf_site_info_group_metrics_B.index.isin(gdf_site_info_group_metrics_A.index)]['site_no'].tolist()
        dict_site_info_avail_years_B = {k:v for k,v in dict_site_info_avail_years_B.items() if k in usgs_sites_B}
        gdf_site_info_group_metrics = pd.concat([gdf_site_info_group_metrics_A,gdf_site_info_group_metrics_B[~gdf_site_info_group_metrics_B.index.isin(gdf_site_info_group_metrics_A.index)]])
        dict_site_info_avail_years = {**dict_site_info_avail_years_A,**dict_site_info_avail_years_B}

    elif frequency == 'hourly':
        gdf_site_info_group_metrics,dict_site_info_avail_years = usgs.filter_stations(gdf_site_info_group_metrics,frequency,[int(start_wy),int(end_wy)],nyears,nid_store_thres,p_thres_dict[frequency])

    gdf_site_info_group_metrics = gpd.sjoin(gdf_site_info_group_metrics,gdf_watersheds,how='left')
    gdf_site_info_group_metrics = gdf_site_info_group_metrics.drop(columns=['index_right','area_m2'])
    
    # Write Availabel Years
    if frequency == 'hourly':
        gdf_site_info_group_metrics['sel_years_HA'] = np.nan
        gdf_site_info_group_metrics['sel_years_HA'] = gdf_site_info_group_metrics['sel_years_HA'].astype(object)
        for i in gdf_site_info_group_metrics.index:
            gdf_site_info_group_metrics.at[i,'sel_years_HA'] = str(dict_site_info_avail_years[gdf_site_info_group_metrics.at[i,'site_no']])
    elif frequency == 'daily':
        gdf_site_info_group_metrics['sel_years_DA'] = np.nan
        gdf_site_info_group_metrics['sel_years_DA'] = gdf_site_info_group_metrics['sel_years_DA'].astype(object)
        for i in gdf_site_info_group_metrics.index:
            gdf_site_info_group_metrics.at[i,'sel_years_DA'] = str(dict_site_info_avail_years[gdf_site_info_group_metrics.at[i,'site_no']])
    elif frequency == 'monthly':
        gdf_site_info_group_metrics['sel_years_MA'] = np.nan
        gdf_site_info_group_metrics['sel_years_MA'] = gdf_site_info_group_metrics['sel_years_MA'].astype(object)
        for i in gdf_site_info_group_metrics.index:
            gdf_site_info_group_metrics.at[i,'sel_years_MA'] = str(dict_site_info_avail_years[gdf_site_info_group_metrics.at[i,'site_no']])
    elif frequency == 'yearly':
        gdf_site_info_group_metrics['sel_years_YA'] = np.nan
        gdf_site_info_group_metrics['sel_years_YA'] = gdf_site_info_group_metrics['sel_years_YA'].astype(object)
        for i in gdf_site_info_group_metrics.index:
            gdf_site_info_group_metrics.at[i,'sel_years_YA'] = str(dict_site_info_avail_years[gdf_site_info_group_metrics.at[i,'site_no']])

    #j=0
    # Calculate Metrics for Each Stations
    for i in gdf_site_info_group_metrics.index:
        #j+=1
        print(i,frequency,period)

        site_id = gdf_site_info.at[i,'site_no']
        station_name = gdf_site_info.at[i,'station_nm']
        nwm3g_comid = gdf_site_info.at[i,'nwm3g_comid']
        nwm3p_comid = gdf_site_info.at[i,'nwm3p_comid']
        if np.isnan(nwm3g_comid):
            comid = int(nwm3p_comid)
        else:
            comid = int(nwm3g_comid)

        # Calibration Basin
        if site_id in list(gdf_nwm21_calib_basins['GAGE_ID']):
            gdf_site_info_group_metrics.at[i,'calib_basin'] = 1
            calib_basin = True
        else:
            gdf_site_info_group_metrics.at[i,'calib_basin'] = 0
            calib_basin = False

        # NWM Link
        if gdf_site_info.at[i,'nwm3g'] == 1:
            nwm_link = True
        else:
            nwm_link = False

        drain_area = gdf_site_info.loc[i,'drain_area_va'] # square miles
        drain_area = drain_area*2.58999 # sq km

        if np.isnan(drain_area):
            gdf_site_info_group_metrics.at[i,'drain_area_km2'] = nwm.get_basin_by_usgsgageid(site_id).to_crs(param_nwm3.crs_nwm_proj4_lcc).area[0]/(1000*1000)
        else:
            gdf_site_info_group_metrics.at[i,'drain_area_km2'] = drain_area

        # Read Data
        df_usgs = pd.read_parquet(os.path.join(usgs_data_dir,frequency,'{}.parquet.gzip'.format(str(site_id))))
        df_nwm = pd.read_parquet(os.path.join(nwm_data_dir,frequency,'{}.parquet.gzip'.format(str(comid))))

        # Apply Start Time/End Time
        df_usgs = usgs.add_water_year(df_usgs)
        df_nwm = usgs.add_water_year(df_nwm)

        df_usgs = df_usgs[(df_usgs['WY']>=dict_site_info_avail_years[site_id][0])&(df_usgs['WY']<=dict_site_info_avail_years[site_id][-1])]
        df_nwm = df_nwm[(df_nwm['WY']>=dict_site_info_avail_years[site_id][0])&(df_nwm['WY']<=dict_site_info_avail_years[site_id][-1])]

        df_usgs = df_usgs.iloc[:,-1:]
        df_nwm = df_nwm.iloc[:,-1:]

        # start_time = '{}-01-01 00:00:00'.format(dict_site_info_avail_years[site_id][0])
        # end_time = '{}-12-31 23:00:00'.format(dict_site_info_avail_years[site_id][-1])
        # df_usgs = df_usgs[start_time:end_time]
        # df_nwm = df_nwm[start_time:end_time]

        Q_nwm_mean = df_nwm.mean().values[0]




        #-------Plot Figures
        figsize=(12,8)
        fig,ax = plt.subplots(2,2,figsize=figsize)
        lw=0.5
        title_fontsize=12
        textbox_fontsize=10

        title_str = ('USGS ID: {} | '.format(str(site_id))+ \
                     'NWM COMID: {} | '.format(str(comid))+ \
                    'Frequency: {} | '.format(str(frequency).capitalize())+ \
                    'Period: {}'.format(str(period[0])+'-'+str(period[1]))+ \
                    '\nName: {} | '.format(str(station_name))+\
                    'Calibration Basin: {} '.format(str(calib_basin)))
        
        fig.suptitle(title_str,fontsize=16)

        # Hydrograph
        df_obs = df_usgs
        df_sim = df_nwm
        df_obs.columns = ['USGS Observation']
        df_sim.columns = ['NWM Simulation']
        df_obs.plot(ax=ax[0,0],style='-k',lw=lw,label='USGS Observation')
        df_sim.plot(ax=ax[0,0],style='--r',lw=lw,label='NWM Simulation')
        ax[0,0].set_ylim(bottom=0)
        ax[0,0].set_ylabel('Discharge (m$^3$/s)')

        
        # Hydrograph (log)
        df_obs = df_usgs
        df_sim = df_nwm
        df_obs.columns = ['USGS Observation']
        df_sim.columns = ['NWM Simulation']

        df_obs.where(df_obs>0,eval_metrics.fill_value).plot(ax=ax[1,0],style='-k',lw=lw,label='USGS Observation')
        df_sim.where(df_sim>0,eval_metrics.fill_value).plot(ax=ax[1,0],style='--r',lw=lw,label='NWM Simulation')
        ax[1,0].set_yscale('log')
        ax[1,0].set_ylabel('Discharge (m$^3$/s)')
        

        # Select Only years with Data above p_thresh
        df_usgs = usgs.add_water_year(df_usgs)
        df_nwm = usgs.add_water_year(df_nwm)
        df_usgs = df_usgs[df_usgs['WY'].isin(dict_site_info_avail_years[site_id])]
        df_nwm = df_nwm[df_nwm['WY'].isin(dict_site_info_avail_years[site_id])]
        df_usgs = df_usgs.iloc[:,-1:]
        df_nwm = df_nwm.iloc[:,-1:]

        # Remove Sites with Constant Values
        if (df_nwm.nunique().values[0] <= 1) == True:
            gdf_site_info_group_metrics.at[i,'nwm_const'] = 1
            #gdf_site_info_group_metrics = gdf_site_info_group_metrics.drop(i)
            #continue
        else:
            gdf_site_info_group_metrics.at[i,'nwm_const'] = 0

        # Writing Mean Q
        gdf_site_info_group_metrics.at[i,'Q_nwm_mean'] = Q_nwm_mean

        #--------


        # Seasonal Analysis (Metrics)
        for season in seasons.keys():
            season_months = seasons[season]['months']
            season_symbol = seasons[season]['symbol']
        
            # Select Months/Seasons
            if season != 'all':
                df_usgs_season = df_usgs[df_usgs.index.month.isin(season_months)]
                df_nwm_season = df_nwm[df_nwm.index.month.isin(season_months)]
            else:
                df_usgs_season = df_usgs
                df_nwm_season = df_nwm


            # Plot Before this
            # Get Corresponding Values from Both Series
            df_usgs_season = df_usgs_season[(df_usgs_season.iloc[:,0].notnull())&
                            (df_nwm_season.iloc[:,0].notnull())]
            df_nwm_season = df_nwm_season[(df_usgs_season.iloc[:,0].notnull())&
                        (df_nwm_season.iloc[:,0].notnull())]

            if ((df_usgs_season.count().values[0]>1)&   # If both df_usgs and df_nwm are not completely empty
                (df_nwm_season.count().values[0]>1)):
                
                # Calculate Metrics
                NSE = eval_metrics.NSE(df_usgs_season,df_nwm_season)
                logNSE = eval_metrics.logNSE(df_usgs_season,df_nwm_season)
                lnNSE = eval_metrics.lnNSE(df_usgs_season,df_nwm_season)
                NNSE = eval_metrics.NNSE(df_usgs_season,df_nwm_season)
                logNNSE = eval_metrics.logNNSE(df_usgs_season,df_nwm_season)
                lnNNSE = eval_metrics.lnNNSE(df_usgs_season,df_nwm_season)
                # KGE = eval_metrics.KGE(df_usgs_season,df_nwm_season)
                # HE_KGE = eval_metrics.HE_KGE(df_usgs_season,df_nwm_season)
                # logKGE = eval_metrics.logKGE(df_usgs_season,df_nwm_season)
                # lnKGE = eval_metrics.lnKGE(df_usgs_season,df_nwm_season)
                PBIAS = eval_metrics.PBIAS(df_usgs_season,df_nwm_season)
                logPBIAS = eval_metrics.logPBIAS(df_usgs_season,df_nwm_season)
                lnPBIAS = eval_metrics.lnPBIAS(df_usgs_season,df_nwm_season)
                PEARSON_R = eval_metrics.PEARSON_R(df_usgs_season,df_nwm_season)
                logPEARSON_R = eval_metrics.logPEARSON_R(df_usgs_season,df_nwm_season)
                lnPEARSON_R = eval_metrics.lnPEARSON_R(df_usgs_season,df_nwm_season)
                MAE = eval_metrics.MAE(df_usgs_season,df_nwm_season)
                nMAE = eval_metrics.nMAE(df_usgs_season,df_nwm_season,drain_area)
                PPD = eval_metrics.PPD(df_usgs_season,df_nwm_season)
                O_AUTOCORR, S_AUTOCORR = eval_metrics.lag_1_autocorr(df_usgs_season,df_nwm_season)
                RMSE = eval_metrics.RMSE(df_usgs_season,df_nwm_season)
                RRMSE = eval_metrics.RRMSE(df_usgs_season,df_nwm_season)

                # Writing Metrics
                gdf_site_info_group_metrics.at[i,'NSE_{}'.format(season_symbol)] = round(NSE,2)
                gdf_site_info_group_metrics.at[i,'logNSE_{}'.format(season_symbol)] = round(logNSE,2)
                gdf_site_info_group_metrics.at[i,'lnNSE_{}'.format(season_symbol)] = round(lnNSE,2)
                gdf_site_info_group_metrics.at[i,'NNSE_{}'.format(season_symbol)] = round(NNSE,2)
                gdf_site_info_group_metrics.at[i,'logNNSE_{}'.format(season_symbol)] = round(logNNSE,2)
                gdf_site_info_group_metrics.at[i,'lnNNSE_{}'.format(season_symbol)] = round(lnNNSE,2)
                # gdf_site_info_group_metrics.at[i,'KGE_{}'.format(season_symbol)] = round(KGE[0],2)
                # gdf_site_info_group_metrics.at[i,'HE_KGE_{}'.format(season_symbol)] = round(HE_KGE[0][0],2)
                # gdf_site_info_group_metrics.at[i,'logKGE_{}'.format(season_symbol)] = round(logKGE[0],2)
                # gdf_site_info_group_metrics.at[i,'lnKGE_{}'.format(season_symbol)] = round(lnKGE[0],2)
                gdf_site_info_group_metrics.at[i,'PBIAS_{}'.format(season_symbol)] = round(PBIAS,2)
                gdf_site_info_group_metrics.at[i,'logPBIAS_{}'.format(season_symbol)] = round(logPBIAS,2)
                gdf_site_info_group_metrics.at[i,'lnPBIAS_{}'.format(season_symbol)] = round(lnPBIAS,2)
                gdf_site_info_group_metrics.at[i,'PEARSON_R_{}'.format(season_symbol)] = round(PEARSON_R,2)
                gdf_site_info_group_metrics.at[i,'logPEARSON_R_{}'.format(season_symbol)] = round(logPEARSON_R,2)
                gdf_site_info_group_metrics.at[i,'lnPEARSON_R_{}'.format(season_symbol)] = round(lnPEARSON_R,2)
                gdf_site_info_group_metrics.at[i,'MAE_{}'.format(season_symbol)] = round(MAE,2)
                gdf_site_info_group_metrics.at[i,'nMAE_{}'.format(season_symbol)] = round(nMAE,2)
                gdf_site_info_group_metrics.at[i,'PPD_{}'.format(season_symbol)] = round(PPD,2)
                gdf_site_info_group_metrics.at[i,'O_AUTOCORR_{}'.format(season_symbol)] = round(O_AUTOCORR,2)
                gdf_site_info_group_metrics.at[i,'S_AUTOCORR_{}'.format(season_symbol)] = round(O_AUTOCORR,2)
                gdf_site_info_group_metrics.at[i,'RMSE_{}'.format(season_symbol)] = round(RMSE,2)
                gdf_site_info_group_metrics.at[i,'RRMSE_{}'.format(season_symbol)] = round(RRMSE,2)



                # Calculate FDC
                
                df_usgs_season_fdc = eval_metrics.FDC(df_usgs_season)
                df_nwm_season_fdc = eval_metrics.FDC(df_nwm_season)

                df_fdc_metrics = eval_metrics.FDC_exceedance_metrics(df_usgs_season,df_nwm_season,exceedances=exceedances)
                df_lnfdc_metrics = eval_metrics.lnFDC_exceedance_metrics(df_usgs_season,df_nwm_season,exceedances=exceedances)

                for exceedance in df_fdc_metrics.index:
                    gdf_site_info_group_metrics.at[i,'PBIAS{}_{}'.format(str(exceedance),season_symbol)] = round(df_fdc_metrics.at[exceedance,'PBIAS'],2)
                    gdf_site_info_group_metrics.at[i,'lnPBIAS{}_{}'.format(str(exceedance),season_symbol)] = round(df_lnfdc_metrics.at[exceedance,'PBIAS'],2)
                    
                    gdf_site_info_group_metrics.at[i,'RMSE{}_{}'.format(str(exceedance),season_symbol)] = round(df_fdc_metrics.at[exceedance,'RMSE'],2)
                    gdf_site_info_group_metrics.at[i,'lnRMSE{}_{}'.format(str(exceedance),season_symbol)] = round(df_lnfdc_metrics.at[exceedance,'RMSE'],2)
                    
                    gdf_site_info_group_metrics.at[i,'RRMSE{}_{}'.format(str(exceedance),season_symbol)] = round(df_fdc_metrics.at[exceedance,'RRMSE'],2)
                    gdf_site_info_group_metrics.at[i,'lnRRMSE{}_{}'.format(str(exceedance),season_symbol)] = round(df_lnfdc_metrics.at[exceedance,'RRMSE'],2)
                    
                    gdf_site_info_group_metrics.at[i,'MPE{}_{}'.format(str(exceedance),season_symbol)] = round(df_fdc_metrics.at[exceedance,'MPE'],2)
                    gdf_site_info_group_metrics.at[i,'lnMPE{}_{}'.format(str(exceedance),season_symbol)] = round(df_lnfdc_metrics.at[exceedance,'MPE'],2)

                    gdf_site_info_group_metrics.at[i,'MPEA{}_{}'.format(str(exceedance),season_symbol)] = round(df_fdc_metrics.at[exceedance,'MPEA'],2)
                    gdf_site_info_group_metrics.at[i,'lnMPEA{}_{}'.format(str(exceedance),season_symbol)] = round(df_lnfdc_metrics.at[exceedance,'MPEA'],2)

                    gdf_site_info_group_metrics.at[i,'MEPE{}_{}'.format(str(exceedance),season_symbol)] = round(df_fdc_metrics.at[exceedance,'MEPE'],2)
                    gdf_site_info_group_metrics.at[i,'lnMEPE{}_{}'.format(str(exceedance),season_symbol)] = round(df_lnfdc_metrics.at[exceedance,'MEPE'],2)

                    gdf_site_info_group_metrics.at[i,'MEPEA{}_{}'.format(str(exceedance),season_symbol)] = round(df_fdc_metrics.at[exceedance,'MEPEA'],2)
                    gdf_site_info_group_metrics.at[i,'lnMEPEA{}_{}'.format(str(exceedance),season_symbol)] = round(df_lnfdc_metrics.at[exceedance,'MEPEA'],2)

                    gdf_site_info_group_metrics.at[i,'VE{}_{}'.format(str(exceedance),season_symbol)] = round(df_fdc_metrics.at[exceedance,'VE'],2)
                    gdf_site_info_group_metrics.at[i,'nVE{}_{}'.format(str(exceedance),season_symbol)] = round(df_fdc_metrics.at[exceedance,'VE']/drain_area,3)

                    gdf_site_info_group_metrics.at[i,'MAPE{}_{}'.format(str(exceedance),season_symbol)] = round(df_lnfdc_metrics.at[exceedance,'MAPE'],2)
                    gdf_site_info_group_metrics.at[i,'lnMAPE{}_{}'.format(str(exceedance),season_symbol)] = round(df_lnfdc_metrics.at[exceedance,'MAPE'],2)
                    
                    gdf_site_info_group_metrics.at[i,'NSE{}_{}'.format(str(exceedance),season_symbol)] = round(df_fdc_metrics.at[exceedance,'NSE'],2)
                    gdf_site_info_group_metrics.at[i,'lnNSE{}_{}'.format(str(exceedance),season_symbol)] = round(df_lnfdc_metrics.at[exceedance,'NSE'],2)
                    
                    gdf_site_info_group_metrics.at[i,'NNSE{}_{}'.format(str(exceedance),season_symbol)] = round(df_fdc_metrics.at[exceedance,'NNSE'],2)
                    gdf_site_info_group_metrics.at[i,'lnNNSE{}_{}'.format(str(exceedance),season_symbol)] = round(df_lnfdc_metrics.at[exceedance,'NNSE'],2)

                # Calculate FDC Yilmaz
                df_fdc_metrics_yilmaz = eval_metrics.FDC_exceedance_metrics_yilmaz(df_usgs_season,df_nwm_season)
                for exceedance in df_fdc_metrics_yilmaz.index:
                    gdf_site_info_group_metrics.at[i,'PBIAS_{}_{}'.format(str(exceedance),season_symbol)] = round(df_fdc_metrics_yilmaz.at[exceedance,'PBIAS'],2)
                    gdf_site_info_group_metrics.at[i,'VE_{}_{}'.format(str(exceedance),season_symbol)] = round(df_fdc_metrics_yilmaz.at[exceedance,'VE'],2)
                    gdf_site_info_group_metrics.at[i,'nVE_{}_{}'.format(str(exceedance),season_symbol)] = round(df_fdc_metrics_yilmaz.at[exceedance,'VE']/drain_area,3)

                # Plot FDCs (Figure 3)
                logy = True
                logx = False
                lw=1.5
                if season == 'all':
                    df_usgs_season_fdc.plot(ax=ax[0,1],style='-k',x='exceedance',y='Q',logy=logy,logx=logx,label='USGS',lw=lw)
                    df_nwm_season_fdc.plot(ax=ax[0,1],style='--k',x='exceedance',y='Q',logy=logy,logx=logx,label='NWM',lw=lw)
                if season == 'summer':
                    df_usgs_season_fdc.plot(ax=ax[0,1],style='-r',x='exceedance',y='Q',logy=logy,logx=logx,label='USGS (Summer)',lw=lw)
                    df_nwm_season_fdc.plot(ax=ax[0,1],style='--r',x='exceedance',y='Q',logy=logy,logx=logx,label='NWM (Summer)',lw=lw)
                if season == 'winter':
                    df_usgs_season_fdc.plot(ax=ax[0,1],style='-b',x='exceedance',y='Q',logy=logy,logx=logx,label='USGS (Winter)',lw=lw)
                    df_nwm_season_fdc.plot(ax=ax[0,1],style='--b',x='exceedance',y='Q',logy=logy,logx=logx,label='NWM (Winter)',lw=lw)
                ax[0,1].set_xlabel('Flow Exceedance (%)')
                ax[0,1].set_ylabel('Discharge (m$^3$/s)')
                ax[0,1].legend(ncols=3)
        
        
        # Add Metrics
        ax[1,1].set_axis_off()

        # # Site Information
        # info_textbox_str = ('Analysis Information:\n'
        #     'USGS ID: {}\n'.format(str(site_id))+ \
        #     'NWM COMID: {} \n'.format(str(comid))+ \
        #     'Frequency: {}\n'.format(str(frequency).capitalize())+ \
        #     'Period: {}\n'.format(str(period[0])+'-'+str(period[1]))
        #     )
        
        # ax[1,1].text(0, 1, info_textbox_str, 
        # transform=ax[1,1].transAxes, 
        # fontsize=textbox_fontsize,
        # verticalalignment='top')

        # Site Metrics


        
        metric_textbox_str = ('Metrics:\n'
            'NNSE: {}\n'.format(str(gdf_site_info_group_metrics.at[i,'NNSE_A']))+ \
            'NSE: {} \n'.format(str(gdf_site_info_group_metrics.at[i,'NSE_A']))+ \
            'PBIAS (%): {} \n'.format(str(gdf_site_info_group_metrics.at[i,'PBIAS_A']))+ \
            'NNSE-ln: {} \n'.format(str(gdf_site_info_group_metrics.at[i,'lnNNSE_A']))+ \
            'NSE-ln: {} \n'.format(str(gdf_site_info_group_metrics.at[i,'lnNSE_A']))+ \
            # 'PBIAS-ln (%): {} \n'.format(str(gdf_site_info_group_metrics.at[i,'lnPBIAS_A']))+ \
            '\n'+\
            'NNSE [Winter]: {}\n'.format(str(gdf_site_info_group_metrics.at[i,'NNSE_W']))+ \
            'NSE [Winter]: {} \n'.format(str(gdf_site_info_group_metrics.at[i,'NSE_W']))+ \
            'PBIAS (%) [Winter]: {} \n'.format(str(gdf_site_info_group_metrics.at[i,'PBIAS_W']))+ \
            'NNSE-ln [Winter]: {} \n'.format(str(gdf_site_info_group_metrics.at[i,'lnNNSE_W']))+ \
            'NSE-ln [Winter]: {} \n'.format(str(gdf_site_info_group_metrics.at[i,'lnNSE_W']))+ \
            # 'PBIAS-ln (%) [Winter]: {} \n'.format(str(gdf_site_info_group_metrics.at[i,'lnPBIAS_W'])) + \
            '\n'+\
            'NNSE [Summer]: {}\n'.format(str(gdf_site_info_group_metrics.at[i,'NNSE_S']))+ \
            'NSE [Summer]: {} \n'.format(str(gdf_site_info_group_metrics.at[i,'NSE_S']))+ \
            'PBIAS (%) [Summer]: {} \n'.format(str(gdf_site_info_group_metrics.at[i,'PBIAS_S']))+ \
            'NNSE-ln [Summer]: {} \n'.format(str(gdf_site_info_group_metrics.at[i,'lnNNSE_S']))+ \
            'NSE-ln [Summer]: {} \n'.format(str(gdf_site_info_group_metrics.at[i,'lnNSE_S'])))
            # 'PBIAS-ln (%) [Summer]: {} \n'.format(str(gdf_site_info_group_metrics.at[i,'lnPBIAS_S']))

            
        ax[1,1].text(0, 1, metric_textbox_str, 
        transform=ax[1,1].transAxes, 
        fontsize=textbox_fontsize,
        verticalalignment='top')


        # FDC Metric
        metric = 'PBIAS'
        metric_label = 'PBIAS'
        metric_textbox_str = 'FDC Metrics:\n'
        
        # Seasons (Yilmaz)
        metric_textbox_str_season = ''
        for season in seasons.keys():
            season_symbol = seasons[season]['symbol']
            metric_textbox_str_season = metric_textbox_str_season+str(season).capitalize()+'\n'
            for exceedance in df_fdc_metrics_yilmaz.index:
                metric_textbox_str_season = metric_textbox_str_season+'{} [{}]: {}\n'.format(metric_label,
                                                            str(exceedance),
                                                            str(gdf_site_info_group_metrics.at[i,'{}_{}_{}'.format(metric,str(exceedance),str(season_symbol))]))
            metric_textbox_str_season = metric_textbox_str_season +'\n'
        metric_textbox_str = metric_textbox_str + metric_textbox_str_season

        ax[1,1].text(0.5, 1, metric_textbox_str, 
        transform=ax[1,1].transAxes, 
        fontsize=textbox_fontsize-1,
        verticalalignment='top')

        # # Seasons
        # metric_textbox_str_season = ''
        # for season in seasons.keys():
        #     season_symbol = seasons[season]['symbol']
        #     metric_textbox_str_season = metric_textbox_str_season+str(season).capitalize()+'\n'
        #     for exceedance in exceedances:
        #         metric_textbox_str_season = metric_textbox_str_season+'{} [Q{}]: {}\n'.format(metric_label,
        #                                                     str(exceedance),
        #                                                     str(gdf_site_info_group_metrics.at[i,'{}{}_{}'.format(metric,str(exceedance),str(season_symbol))]))
        #     metric_textbox_str_season = metric_textbox_str_season +'\n'
        # metric_textbox_str = metric_textbox_str + metric_textbox_str_season

        # ax[1,1].text(0.5, 1, metric_textbox_str, 
        # transform=ax[1,1].transAxes, 
        # fontsize=textbox_fontsize-1,
        # verticalalignment='top')

        title_str = ('USGS ID: {} | '.format(str(site_id))+ \
                'NWM COMID: {} | '.format(str(comid))+ \
                'Area: {} km$^2$ | '.format(str(round(drain_area,2)))+ \
            'Frequency: {} | '.format(str(frequency).capitalize())+ \
            'Period: {}'.format(str(period[0])+'-'+str(period[1]))+ \
            '\nName: {} | '.format(str(station_name))+\
            'Calibration Basin: {} | '.format(str(calib_basin))+\
            'Basin: {} | '.format(gdf_site_info_group_metrics.at[i,'NAME_ABR'])+\
            'NWM Link: {}'.format(str(nwm_link)))
        
        fig.suptitle(title_str,fontsize=14)
        plt.tight_layout()

        basin_abr = gdf_site_info_group_metrics.at[i,'NAME_ABR']
        misc.makedir(os.path.join(savedir_fig,frequency))
        fig.savefig(os.path.join(savedir_fig,frequency,'{}_{}.png'.format(basin_abr,str(site_id))),dpi=300,bbox_inches='tight')
        #print(j,site_id,station_name)
    # gdf_site_info_group_metrics.to_file()

    # Save Files
    gdf_site_info_group_metrics['nid_ID'] = gdf_site_info_group_metrics['nid_ID'].astype(str)
    gdf_site_info_group_metrics.to_file(os.path.join(savedir_gpkg,'{}_{}_{}_utm12n_nad83.gpkg'.format(str(frequency),str(period[0]),str(period[1]))))
    gdf_site_info_group_metrics.to_file(os.path.join(savedir_shp,'{}_{}_{}_utm12n_nad83.shp'.format(str(frequency),str(period[0]),str(period[1]))))
    gdf_site_info_group_metrics.to_pickle(os.path.join(savedir_pkl,'{}_{}_{}_utm12n_nad83.pkl'.format(str(frequency),str(period[0]),str(period[1]))))
    gdf_site_info_group_metrics.to_csv(os.path.join(savedir_csv,'{}_{}_{}_utm12n_nad83.csv'.format(str(frequency),str(period[0]),str(period[1]))))
    gdf_site_info_group_metrics.to_parquet(os.path.join(savedir_parquet,'{}_{}_{}_utm12n_nad83.parquet.gzip'.format(str(frequency),str(period[0]),str(period[1]))))
