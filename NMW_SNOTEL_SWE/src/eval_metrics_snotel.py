import pandas as pd
import numpy as np
from scipy.stats import pearsonr
import hydroeval as he
import datetime
import calendar
import matplotlib.pyplot as plt

# fill_value = 3E-05
fill_value = 1E-06

def days_in_wy(year):
    return 365 + calendar.isleap(year)

def hours_in_wy(year):
    return days_in_wy(year) * 24

def months_in_wy(year):
    return 12

def data_availability(df,p_thres,min_wy_thres):
    df = add_water_year(df)
    df_avail = df.groupby('WY').count().iloc[:,1:]
    df_max_avail = pd.Series(df_avail.index)
    df_max_avail = df_max_avail.apply(days_in_wy)
    df_max_avail.index = df_avail.index
    df_avail = df_avail.apply(lambda x: x / df_max_avail)
    df_avail = df_avail*100
    df_thres_count = df_avail[df_avail>=p_thres].count()
    df_thres_count_stations = df_thres_count[df_thres_count>=min_wy_thres]
    return {'p_avail':df_avail,'avail_wy_thres':df_thres_count,'avail_wy_thres_stations':list(df_thres_count_stations.index)}

def filter_by_avail(df,p_thres,min_wy_thres):
    return df[data_availability(df,p_thres,min_wy_thres)['avail_wy_thres_stations']]

def add_water_year(df):
    df['WY'] = df.index.year.where(df.index.month < 10, df.index.year + 1)
    df = pd.concat([df.iloc[:,-1:],df.iloc[:,:-1]],axis=1)
    return df

def process_df(df,start_wy,end_wy):
    df.index = df.index.tz_localize(None)
    df = add_water_year(df)
    df = df.loc[(df['WY']>=start_wy) & (df['WY']<=end_wy)]
    return df

def groupby_wy_mean(df):
    df = add_water_year(df)
    return df.groupby('WY').mean()

def groupby_wy_sum(df):
    df = add_water_year(df)
    return df.groupby('WY').sum()


def preprocess_df_metrics(obs,sim):
    global fill_value
    obs = obs[(obs.notnull()) & (sim.notnull())]
    sim = sim[(obs.notnull()) & (sim.notnull())]

    obs = obs.where(obs > 0.0, fill_value)
    sim = sim.where(sim > 0.0, fill_value)

    return obs,sim

def calc_metric_stats(df):

    # Mean
    df_mean = df.groupby(by=[df.index.month,df.index.day]).mean()
    df_mean.index.names = ['month','day']
    df_mean = df_mean.reindex([10,11,12,1,2,3,4,5,6,7,8,9],level=0)
    df_mean = df_mean.drop((2,29)) # Drop Feb 29
    df_mean.index = pd.date_range(start=f'2000-10-01',end=f'2001-09-30',freq='D')
    df_mean.name = 'mean'

    # Max
    df_max = df.groupby(by=[df.index.month,df.index.day]).max()
    df_max.index.names = ['month','day']
    df_max = df_max.reindex([10,11,12,1,2,3,4,5,6,7,8,9],level=0)
    df_max = df_max.drop((2,29))
    df_max.index = pd.date_range(start=f'2000-10-01',end=f'2001-09-30',freq='D')
    df_max.name = 'max'

    # Min
    df_min = df.groupby(by=[df.index.month,df.index.day]).min()
    df_min.index.names = ['month','day']
    df_min = df_min.reindex([10,11,12,1,2,3,4,5,6,7,8,9],level=0)
    df_min = df_min.drop((2,29))
    df_min.index = pd.date_range(start=f'2000-10-01',end=f'2001-09-30',freq='D')
    df_min.name = 'min'

    # 5% Percentile
    df_5 = df.groupby(by=[df.index.month,df.index.day]).quantile(0.05)
    df_5.index.names = ['month','day']
    df_5 = df_5.reindex([10,11,12,1,2,3,4,5,6,7,8,9],level=0)
    df_5 = df_5.drop((2,29))
    df_5.index = pd.date_range(start=f'2000-10-01',end=f'2001-09-30',freq='D')
    df_5.name = 'P5'

    # 95% Percentile
    df_95 = df.groupby(by=[df.index.month,df.index.day]).quantile(0.95)
    df_95.index.names = ['month','day']
    df_95 = df_95.reindex([10,11,12,1,2,3,4,5,6,7,8,9],level=0)
    df_95 = df_95.drop((2,29))
    df_95.index = pd.date_range(start=f'2000-10-01',end=f'2001-09-30',freq='D')
    df_95.name = 'P95'
    
    df = pd.concat([df_mean,df_max,df_min,df_5,df_95],axis=1)
    return df

def get_static_nwm_data(data,gdf_sites,var_name):
    dfs = {}
    for i in gdf_sites.index:
        site_code = gdf_sites.loc[i, 'code']
        site_name = gdf_sites.loc[i, 'name']
        site_x = gdf_sites.loc[i, 'geometry'].x
        site_y = gdf_sites.loc[i, 'geometry'].y
        if var_name == 'HGT':
            df = float(data.sel(x=site_x, y=site_y, method='nearest').values)
        if var_name == 'ISLTYP':
            df = int(data.sel(x=site_x, y=site_y, method='nearest').values)
        if var_name == 'IVGTYP':
            df = int(data.sel(x=site_x, y=site_y, method='nearest').values)
        dfs[site_code] = df
    dfs = pd.Series(dfs)
    dfs.name = var_name
    dfs = dfs.to_frame()
    return dfs


def NSE(obs,sim):
    obs,sim = preprocess_df_metrics(obs,sim)
    numerator = ((obs - sim)**2).sum()
    denominator = ((obs - obs.mean())**2).sum()
    nse = 1-(numerator/denominator)
    nse = nse.where(denominator != 0.0, -np.inf)
    return nse

def NNSE(obs,sim):
    nse = NSE(obs,sim)
    nnse = 1/(2-nse)
    return nnse

def RMSE(obs,sim):
    obs,sim = preprocess_df_metrics(obs,sim)
    rmse = np.sqrt(((obs - sim)**2).mean())
    return rmse

def PBIAS(obs,sim):
    obs,sim = preprocess_df_metrics(obs,sim)
    pbias = ((sim - obs).sum() / obs.sum()) * 100
    return pbias

def BIAS(obs,sim):
    obs,sim = preprocess_df_metrics(obs,sim)
    bias = (sim - obs).mean()
    return bias

def BIAS_metrics(obs,sim):
    obs,sim = preprocess_df_metrics(obs,sim)
    bias = (sim - obs)
    return calc_metric_stats(bias)

    # if (denominator == 0):
    #     nse = -np.inf
    # else:
    #     nse = 1-(numerator/denominator)

    # numerator = ((df['Observed'] - df['Simulated'])**2).sum()
    # denominator = ((df['Observed'] - df['Observed'].mean())**2).sum()
    # if (denominator == 0):
    #     nse = -np.inf
    # else:
    #     nse = 1-(numerator/denominator)
    # #     print('Warning: Zero Denominator')
    # #     denominator = 1E-06
    # # nse = 1-(numerator/denominator)
    # return nse

def calc_nstation_availability_by_threshold(df, p_thresholds, start_wy, end_wy):
    df = process_df(df,start_wy,end_wy)
    dfs_avail = pd.DataFrame(index=range(1,end_wy-start_wy+2),
                                columns=['p={}%'.format(p) for p in p_thresholds])
    for nyears in dfs_avail.index:
        for p_threshold in p_thresholds:
            df_avail = data_availability(df,p_threshold,nyears)
            nstations = len(df_avail['avail_wy_thres_stations'])
            dfs_avail.at[nyears,'p={}%'.format(p_threshold)] = nstations
    return dfs_avail

def fice_jordan1991(tsfc,tfrz):
    if (tsfc < tfrz+0.5):
        return 1.0
    elif (tfrz+0.5 <= tsfc) & (tsfc < tfrz+2.0):
        return 1.0-(-54.632+0.2*tsfc)
    elif (tfrz+2.0 <= tsfc) & (tsfc < tfrz+2.5):
        return 0.6
    else:
        return 0.0


if __name__ == "__main__":
    tfrz = 273.15 #K
    tsfc = 273.15 #K
    fice = fice_jordan1991(tfrz,tsfc)

    tsfc_arr = np.arange(272.15, 276.15, 0.1)
    fice_arr = np.array([fice_jordan1991(tfrz, tsfc) for tsfc in tsfc_arr])

    fig,ax = plt.subplots()
    plt.plot(tsfc_arr,fice_arr)
