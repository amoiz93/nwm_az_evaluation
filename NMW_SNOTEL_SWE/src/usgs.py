import os
import numpy as np
import pandas as pd
import geopandas as gpd
import misc
import dataretrieval.nwis as nwis
import pyproj

def download_usgs_inst_discharge(site_id):
    df = nwis.get_record(sites=site_id, 
                        service='iv',
                        start='1900-01-01',
                        parameterCd='00060')
    return df

def download_usgs_daily_discharge(site_id):
    df = nwis.get_record(sites=site_id, 
                        service='dv',
                        start='1900-01-01',
                        parameterCd='00060')
    return df

def get_usgs_site_info_by_shapefile(states,bounding_gdf,from_crs,parameterCd):
    usgs_sites = []
    for state in states:
        usgs_sites_state = nwis.get_info(stateCd=state,parameterCd=parameterCd)[0]
        usgs_sites.append(usgs_sites_state)
    usgs_sites = pd.concat(usgs_sites,ignore_index=True)
    transformer = pyproj.Transformer.from_crs(from_crs, bounding_gdf.crs)
    for i in usgs_sites.index:
        x, y = transformer.transform(usgs_sites.iloc[i]['dec_lat_va'],usgs_sites.iloc[i]['dec_long_va'])
        usgs_sites.loc[i,'x'] = x
        usgs_sites.loc[i,'y'] = y
    gdf_usgs_sites = gpd.GeoDataFrame(usgs_sites, 
                                      geometry=gpd.points_from_xy(x=usgs_sites.x, y=usgs_sites.y,
                                      crs=bounding_gdf.crs))
    gdf_usgs_sites = gpd.clip(gdf_usgs_sites,bounding_gdf)
    gdf_usgs_sites = gdf_usgs_sites.reset_index()
    return gdf_usgs_sites

def check_headers_daily_discharge(df):
    site_id = df['site_no'].iloc[0]
    
    # Processing special columns in USGS gage data

    # Case 1
    if misc.unordered_lists_are_equal(list(df.columns.values),
                               ['00060_data prior to 10/1/1992_Mean',
                                '00060_data prior to 10/1/1992_Mean_cd',
                                'site_no',
                                '00060_data from 10/1/1992 forward_Mean',
                                '00060_data from 10/1/1992 forward_Mean_cd']):
        df['00060_data from 10/1/1992 forward_Mean'] = df[['00060_data prior to 10/1/1992_Mean',
                                                            '00060_data from 10/1/1992 forward_Mean']].mean(axis=1)
        df['00060_data from 10/1/1992 forward_Mean_cd'] = df['00060_data from 10/1/1992 forward_Mean_cd'].fillna(df['00060_data prior to 10/1/1992_Mean_cd'])
        df = df.drop(columns=['00060_data prior to 10/1/1992_Mean',
                                                              '00060_data prior to 10/1/1992_Mean_cd'])
        df = df.rename(columns={'00060_data from 10/1/1992 forward_Mean':'00060_Mean',
                                '00060_data from 10/1/1992 forward_Mean_cd':'00060_Mean_cd'})
    
    # Case 2
    elif misc.unordered_lists_are_equal(list(df.columns.values),
                                   ['00060_elevation:      2525._Mean',
                                    '00060_elevation:      2525._Mean_cd',
                                    'site_no']):
        df = df.rename(columns={'00060_elevation:      2525._Mean':'00060_Mean',
                                '00060_elevation:      2525._Mean_cd':'00060_Mean_cd'})
    
    # Case 3
    elif misc.unordered_lists_are_equal(list(df.columns.values),
                                   ['00060_elevation:      5143._Mean',
                                    '00060_elevation:      5143._Mean_cd',
                                    'site_no']):
        df = df.rename(columns={'00060_elevation:      5143._Mean':'00060_Mean',
                                '00060_elevation:      5143._Mean_cd':'00060_Mean_cd'})
    
    # Case 4
    elif misc.unordered_lists_are_equal(list(df.columns.values),
                                   ['00060_2_Mean',
                                    '00060_2_Mean_cd',
                                    'site_no',
                                    '00060_Mean',
                                    '00060_Mean_cd']):
        df['00060_2_Mean'] = df[['00060_2_Mean',
                                 '00060_Mean']].mean(axis=1)
        df['00060_2_Mean_cd'] = df['00060_2_Mean_cd'].fillna(df['00060_Mean_cd'])
        df = df.drop(columns=['00060_Mean',
                              '00060_Mean_cd'])
        df = df.rename(columns={'00060_2_Mean':'00060_Mean',
                                '00060_2_Mean_cd':'00060_Mean_cd'})
    
    # Case 5
    elif misc.unordered_lists_are_equal(list(df.columns.values),
                                   ['00060_Mean',
                                    '00060_Mean_cd',
                                    'site_no',
                                    '00060_Observation at 24:00',
                                    '00060_Observation at 24:00_cd']):
        df = df.drop(columns=['00060_Observation at 24:00',
                              '00060_Observation at 24:00_cd'])

    # Case 6
    elif misc.unordered_lists_are_equal(list(df.columns.values),
                                   ['00060_2_Mean',
                                    '00060_2_Mean_cd',
                                    'site_no',
                                    '00060_Sum',
                                    '00060_Sum_cd']):
        df = df.drop(columns=['00060_Sum',
                              '00060_Sum_cd'])
        df = df.rename(columns={'00060_2_Mean':'00060_Mean',
                                '00060_2_Mean_cd':'00060_Mean_cd'})
        
    # Case 7
    elif misc.unordered_lists_are_equal(list(df.columns.values),
                                   ['00060_all flows above imperial dam_Mean',
                                   '00060_all flows above imperial dam_Mean_cd',
                                    'site_no']):
        df = df.rename(columns={'00060_all flows above imperial dam_Mean':'00060_Mean',
                               '00060_all flows above imperial dam_Mean_cd':'00060_Mean_cd'})
    
    # Case 8
    elif pd.DataFrame(['site_no','00060_Mean','00060_Mean_cd']).isin(list(df.columns.values)).all().values:
        df = df[['site_no',
                 '00060_Mean',
                 '00060_Mean_cd']]
    
    # Final Check
    if not misc.unordered_lists_are_equal(list(df.columns.values), 
                                 ['00060_Mean','00060_Mean_cd','site_no']):
        raise IndexError('''Site {} DataFrame has an incorrect number of columns or incorrect column names.
                          DataFrame has columns {}.
                          Columns must be {}.'''.format(site_id,str(df.columns.values),str(str(['00060_Mean','00060_Mean_cd','site_no']))))

    return df

def check_headers_inst_discharge(df):
    site_id = df['site_no'].iloc[0]
    
    # Processing special columns in USGS gage data
    
    # Case 1
    if misc.unordered_lists_are_equal(list(df.columns.values),
                                 ['00060_2',
                                  '00060_2_cd',
                                  'site_no',
                                  '00060',
                                  '00060_cd']):
        df['00060_2'] = df[['00060','00060_2']].mean(axis=1)
        df['00060_2_cd'] = df['00060_2_cd'].fillna(df['00060_cd'])
        df = df.drop(columns=['00060','00060_cd'])
        df = df.rename(columns={'00060_2':'00060',
                                '00060_2_cd':'00060_cd'})
    
    # Case 2
    elif misc.unordered_lists_are_equal(list(df.columns.values),
                                   ['00060_discharge',
                                    '00060_discharge_cd',
                                    'site_no']):
        df = df.rename(columns={'00060_discharge':'00060',
                                '00060_discharge_cd':'00060_cd'})
    
    # Case 3
    elif misc.unordered_lists_are_equal(list(df.columns.values),
                                    ['00060_2',
                                     '00060_2_cd',
                                     'site_no']):
        df = df.rename(columns={'00060_2':'00060',
                                '00060_2_cd':'00060_cd'})
    
    # Case 4
    elif misc.unordered_lists_are_equal(list(df.columns.values),
                                   ['00060_bridge location',
                                     '00060_bridge location_cd',
                                     'site_no']):
        df = df.rename(columns={'00060_bridge location':'00060',
                                '00060_bridge location_cd':'00060_cd'})
    
    # Case 5
    elif misc.unordered_lists_are_equal(list(df.columns.values),
                                    ['00060',
                                     '00060_cd',
                                     'site_no',
                                     '00060_outlet pipe',
                                     '00060_outlet pipe_cd']):
        df = df.drop(columns=['00060_outlet pipe',
                              '00060_outlet pipe_cd'])
    
    # Case 6
    elif misc.unordered_lists_are_equal(list(df.columns.values),
                                    ['00060_historic 10/1987 to 09/2006',
                                     '00060_historic 10/1987 to 09/2006_cd',
                                     'site_no']):
        df = df.rename(columns={'00060_historic 10/1987 to 09/2006':'00060',
                                '00060_historic 10/1987 to 09/2006_cd':'00060_cd'}) 
    
    # Case 7
    elif misc.unordered_lists_are_equal(list(df.columns.values),
                                    ['00060_primary stage',
                                     '00060_primary stage_cd',
                                     'site_no']):
        df = df.rename(columns={'00060_primary stage':'00060',
                                '00060_primary stage_cd':'00060_cd'})

    # Final Check
    if not misc.unordered_lists_are_equal(list(df.columns.values), 
                                 ['00060','00060_cd','site_no']):
        raise IndexError('''Site {} DataFrame has an incorrect number of columns or incorrect column names.
                          DataFrame has columns {}.
                          Columns must be {}.'''.format(site_id,str(df.columns.values),str(str(['00060_Mean','00060_Mean_cd','site_no']))))

    return df
    
def calc_missing_data_stat(df):
    water_years = list(df.index.year.unique()+1)
    df_stat = pd.DataFrame(index=water_years,columns=['n_missing','n_available','n_negative','min_negative','n_total','percentage_missing','percentage_available','percentage_negative'])
    freq = df.index.freq
    for water_year in water_years:
        # Get Water Year Data (depending upin data frequency the endpoint might differ)
        if freq == '15T':
            df_water_year = df['{}-10-01 00:00:00'.format(str(water_year-1)):'{}-09-30 23:45:00'.format(str(water_year))]
        elif freq == '1H':
            df_water_year = df['{}-10-01 00:00:00'.format(str(water_year-1)):'{}-09-30 23:00:00'.format(str(water_year))]
        elif freq == '1D':
            df_water_year = df['{}-10-01'.format(str(water_year-1)):'{}-09-30'.format(str(water_year))]
        
        # Calculate data availability percentages
        n_total = len(df_water_year['Q_cfs'])
        
        if n_total != 0:
            n_missing = df_water_year['Q_cfs'].isna().sum()
            n_available = df_water_year['Q_cfs'].count()
            n_negative = df_water_year[df_water_year['Q_cfs']<0]['Q_cfs'].count()
            min_negative = df_water_year[df_water_year['Q_cfs']<0]['Q_cfs'].min()
            
            
            percentage_missing = (n_missing/n_total)*100
            percentage_available = (n_available/n_total)*100
            percentage_negative = (n_negative/n_total)*100

            df_stat.loc[water_year,'n_missing'] = n_missing
            df_stat.loc[water_year,'n_available'] = n_available
            df_stat.loc[water_year,'n_negative'] = n_negative
            df_stat.loc[water_year,'min_negative'] = min_negative
            df_stat.loc[water_year,'n_total'] = n_total
            df_stat.loc[water_year,'percentage_missing'] = percentage_missing
            df_stat.loc[water_year,'percentage_available'] = percentage_available
            df_stat.loc[water_year,'percentage_negative'] = percentage_negative
        df_stat = df_stat[~df_stat['n_total'].isna()]
    return df_stat

def write_missing_data_stat(df_n_missing,
                            df_n_available,
                            df_n_negative,
                            df_min_negative,
                            df_n_total,
                            df_percentage_missing,
                            df_percentage_available,
                            df_percentage_negative,
                            savedir):
    # Remove Dummy Years (Water Years with NaN values)
    df_n_missing = df_n_missing.dropna(how='all')
    df_n_available = df_n_available.dropna(how='all')
    df_n_negative = df_n_negative.dropna(how='all')
    df_min_negative = df_min_negative.dropna(how='all')
    df_n_total = df_n_total.dropna(how='all')
    df_percentage_missing = df_percentage_missing.dropna(how='all')
    df_percentage_available = df_percentage_available.dropna(how='all')
    df_percentage_negative = df_percentage_negative.dropna(how='all')

    # Writing Data Availability Statistics (.csv)
    df_n_missing.to_csv(os.path.join(savedir,'info','n_missing.txt'))
    df_n_available.to_csv(os.path.join(savedir,'info','n_available.txt'))
    df_n_negative.to_csv(os.path.join(savedir,'info','n_negative.txt'))
    df_min_negative.to_csv(os.path.join(savedir,'info','min_negative.txt'))
    df_n_total.to_csv(os.path.join(savedir,'info','n_total.txt'))
    df_percentage_missing.to_csv(os.path.join(savedir,'info','percentage_missing.txt'))
    df_percentage_available.to_csv(os.path.join(savedir,'info','percentage_available.txt'))
    df_percentage_negative.to_csv(os.path.join(savedir,'info','percentage_negative.txt'))

    # Writing Data Availability Statistics (.pkl)
    df_n_missing.to_pickle(os.path.join(savedir,'info','n_missing.pkl'))
    df_n_available.to_pickle(os.path.join(savedir,'info','n_available.pkl'))
    df_n_negative.to_pickle(os.path.join(savedir,'info','n_negative.pkl'))
    df_min_negative.to_pickle(os.path.join(savedir,'info','min_negative.pkl'))
    df_n_total.to_pickle(os.path.join(savedir,'info','n_total.pkl'))
    df_percentage_missing.to_pickle(os.path.join(savedir,'info','percentage_missing.pkl'))
    df_percentage_available.to_pickle(os.path.join(savedir,'info','percentage_available.pkl'))
    df_percentage_negative.to_pickle(os.path.join(savedir,'info','percentage_negative.pkl'))


def check_percentage_availability(df_usgs_obs,start_dt,end_dt,freq):
    # Filter Values
    df_usgs_obs_filtered = df_usgs_obs[start_dt:end_dt]
    n_avail = df_usgs_obs_filtered.count()
    n_total = len(pd.date_range(start_dt,end_dt,freq=freq))
    p_avail = (n_avail/n_total)*100
    return p_avail

def check_percentage_availability_wy(df_usgs_obs,wy):
    # Filter Values
    df_usgs_obs_filtered = df_usgs_obs[(df_usgs_obs['WY']==wy)]
    n_avail = df_usgs_obs_filtered.iloc[:,-1].count()
    n_total = df_usgs_obs_filtered.iloc[:,0].count()
    p_avail = (n_avail/n_total)*100
    return p_avail

def add_water_year(df):
    df['WY'] = df.index.year.where(df.index.month < 10, df.index.year + 1)
    df = pd.concat([df.iloc[:,-1:],df.iloc[:,:-1]],axis=1)
    return df

# def calc_ngage_availability_by_threshold(gdf_site_info,freq,start_year,end_year,p_thresholds):
#     if freq == 'inst':
#         gdf_site_info = gdf_site_info[gdf_site_info['inst']==1]
#     elif freq == 'daily':
#         gdf_site_info = gdf_site_info[gdf_site_info['daily']==1]
    
#     avail_labels = []
#     for year in range(start_year,end_year+1):
#         if freq == 'inst':
#             avail_labels.append('HA{}'.format(str(year)))
#         elif freq == 'daily':
#             avail_labels.append('DA{}'.format(str(year)))
        
#     gdf_site_info = gdf_site_info[avail_labels]
#     df_availability = []
#     for p_threshold in p_thresholds:
#         gdf_site_info_thres = gdf_site_info[gdf_site_info>=p_threshold]
#         gdf_site_info_count = gdf_site_info_thres.count(axis=1)
#         gdf_site_info_hist = np.histogram(gdf_site_info_count,bins=range(0,gdf_site_info_count.max()+2,1))
#         df_site_info_hist = pd.DataFrame(index=gdf_site_info_hist[1][0:-1],data=gdf_site_info_hist[0])
#         df_site_info_hist = df_site_info_hist.iloc[::-1]
#         df_site_info_hist = df_site_info_hist.cumsum()
#         df_site_info_hist.columns = ['p={}%'.format(str(p_threshold))]
#         df_availability.append(df_site_info_hist)
#     df_availability = pd.concat(df_availability,axis=1)
#     df_availability.index.name = 'n_years'
#     df_availability = df_availability.where(df_availability.notnull(),0)
#     #df_availability = df_availability[df_availability.columns[::-1]]
#     return df_availability

# def calc_ngage_availability_by_threshold_end_year(gdf_site_info,freq,start_year,end_year,p_thresholds):
#     if freq == 'inst':
#         freq_label='HA'
#         gdf_site_info = gdf_site_info[gdf_site_info['inst']==1]
#     elif freq == 'daily':
#         freq_label='DA'
#         gdf_site_info = gdf_site_info[gdf_site_info['daily']==1]

#     avail_labels = []
#     for year in range(start_year,end_year+1):
#         avail_labels.append('{}{}'.format(freq_label,str(year)))
        
#     gdf_site_info = gdf_site_info[avail_labels]
#     df_availability = []
#     for p_threshold in p_thresholds:
#         print(p_threshold)
#         gdf_site_info_thres = gdf_site_info[gdf_site_info>=p_threshold]
#         gdf_site_info_thres = gdf_site_info_thres[gdf_site_info_thres.columns[::-1]]
#         gdf_site_info_minyear = gdf_site_info_thres.cumsum(axis=1,skipna=False).idxmax(axis=1)
#         gdf_site_info_count = end_year+1-(gdf_site_info_minyear.dropna().replace(freq_label,'',regex=True)).astype('int')
#         gdf_site_info_hist = np.histogram(gdf_site_info_count,bins=range(0,gdf_site_info_count.max()+2,1))
#         df_site_info_hist = pd.DataFrame(index=gdf_site_info_hist[1][0:-1],data=gdf_site_info_hist[0])
#         df_site_info_hist = df_site_info_hist.iloc[::-1]
#         df_site_info_hist = df_site_info_hist.cumsum()
#         df_site_info_hist.columns = ['p={}%'.format(str(p_threshold))]
#         df_availability.append(df_site_info_hist)
#     df_availability = pd.concat(df_availability,axis=1)
#     df_availability.index.name = 'n_years'
#     df_availability = df_availability.where(df_availability.notnull(),0)
#     #df_availability = df_availability[df_availability.columns[::-1]]
#     return df_availability

# Get List of Gages with Data Available >= p% between start/end year
# This fuction might be redundant (remove later)
def list_of_gages_with_pavail_start_end_year(gdf_site_info,frequency,start_year,end_year,p_threshold):
    gdf_site_info = gdf_site_info.copy()
    gdf_site_info.index = gdf_site_info['site_no']
    if frequency == 'hourly':
        gdf_site_info = gdf_site_info[gdf_site_info['inst']==1]
    elif frequency == 'daily':
        gdf_site_info = gdf_site_info[gdf_site_info['daily']==1]

    avail_labels = []
    for year in range(start_year,end_year+1):
        if frequency == 'hourly':
            avail_labels.append('HA{}'.format(str(year)))
        elif frequency == 'daily':
            avail_labels.append('DA{}'.format(str(year)))
    
    gdf_site_info = gdf_site_info[avail_labels]

    gdf_site_info_thres = gdf_site_info[gdf_site_info>=p_threshold]
    gdf_site_info_count = gdf_site_info_thres.count(axis=1)
    gdf_site_info_count = gdf_site_info_count[gdf_site_info_count>=len(range(start_year,end_year+1))]
    site_years = {}
    for site in gdf_site_info_count.index:
        year_list = (list(gdf_site_info_thres.loc[site,:].dropna().index))
        year_list = [int(x[2:]) for x in year_list]
        site_years[site] = year_list
    return site_years
    
# Get List of Gages with Data Available >= p% in any n years
def list_of_gages_with_pavail_nyear(gdf_site_info,frequency,p_threshold,nyears):
    gdf_site_info = gdf_site_info.copy()
    gdf_site_info.index = gdf_site_info['site_no']
    if frequency == 'hourly':
        gdf_site_info = gdf_site_info[gdf_site_info['inst']==1]
    elif frequency == 'daily':
        gdf_site_info = gdf_site_info[gdf_site_info['daily']==1]

    avail_labels = []
    for year in range(1980,2022+1):
        if frequency == 'hourly':
            avail_labels.append('HA{}'.format(str(year)))
        elif frequency == 'daily':
            avail_labels.append('DA{}'.format(str(year)))
    
    gdf_site_info = gdf_site_info[avail_labels]

    gdf_site_info_thres = gdf_site_info[gdf_site_info>=p_threshold]
    gdf_site_info_count = gdf_site_info_thres.count(axis=1)
    gdf_site_info_count = gdf_site_info_count[gdf_site_info_count>=nyears]
    
    site_years = {}
    for site in gdf_site_info_count.index:
        year_list = (list(gdf_site_info_thres.loc[site,:].dropna().index))
        year_list = [int(x[2:]) for x in year_list]
        site_years[site] = year_list
    return site_years

def list_of_gages_with_pavail(gdf_site_info,frequency,start_year,end_year,nyears,p_threshold):
    list_start_end_year = list_of_gages_with_pavail_start_end_year(gdf_site_info,frequency,start_year,end_year,p_threshold)
    list_nyear = list_of_gages_with_pavail_nyear(gdf_site_info,frequency,p_threshold,nyears)
    list_pavail = list(set(list_start_end_year.keys()) | set(list_nyear.keys()))

    # Combine both lists
    site_years = {}
    for site in list_pavail:
        if site in list_start_end_year.keys():
            site_years[site] = list_start_end_year[site]
        elif site in list_nyear.keys():
            site_years[site] = list_nyear[site]

    site_latest_nyears = {}
    for site in site_years.keys():
        site_latest_nyears[site] = sorted(site_years[site])[-1*nyears:]
    return site_latest_nyears

def calc_ngage_availability_by_threshold(gdf_site_info,frequency,p_thresholds):
    df_avail = pd.DataFrame(index=range(1,44),
                              columns=['p={}%'.format(p) for p in p_thresholds])
    for nyears in df_avail.index:
        for p_avail in p_thresholds:
            avail = list_of_gages_with_pavail_nyear(gdf_site_info,frequency,p_avail,nyears)
            df_avail.at[nyears,'p={}%'.format(p_avail)] = len(avail.keys())
    return df_avail

# Filter stations based on certain conditions
def filter_stations(gdf_site_info,frequency,period,nyears,nid_store_thres,p_thres):
    gdf_site_info = gdf_site_info.copy()
    # Remove sites at Yuma
    gdf_site_info = gdf_site_info[gdf_site_info['yuma']==0] # Exclude YUMA stations

    # # Remove Sites with Constant NWM Discharge
    # gdf_site_info = gdf_site_info[(gdf_site_info['nwm_const']!=1)|
    #                             (gdf_site_info['nwm_const_sel']!=1)]

    # Check NID Storage and HCDN2009
    gdf_site_info = gdf_site_info[(gdf_site_info['nid_maxstorage']<nid_store_thres)|
                                (gdf_site_info['hcdn2009']==1)]

    # Check Availability of Data
    if frequency in ['daily','monthly','yearly']:
        gdf_site_info = gdf_site_info[gdf_site_info['daily']==1]
        selected_gages = list_of_gages_with_pavail(gdf_site_info,'daily',period[0],period[1],nyears,p_thres)
    elif frequency == 'hourly':
        gdf_site_info = gdf_site_info[gdf_site_info['inst']==1]
        selected_gages = list_of_gages_with_pavail(gdf_site_info,'hourly',period[0],period[1],nyears,p_thres)

    gdf_site_info = gdf_site_info[gdf_site_info['site_no'].isin(list(selected_gages.keys()))]

    return gdf_site_info, selected_gages

