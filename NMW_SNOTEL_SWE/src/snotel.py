import ulmo
import pandas as pd
import numpy as np
import datetime
from shapely.geometry import Point
import geopandas as gpd

def add_water_year(df):
    df['WY'] = df.index.year.where(df.index.month < 10, df.index.year + 1)
    df = pd.concat([df.iloc[:,-1:],df.iloc[:,:-1]],axis=1)
    return df

def snotel_huc12_to_huc8(huc12):
    return huc12[:-4]

def snotel_site_comments(site_comments):
    out_col = {}
    for col in site_comments.split('|'):
        col_name = col.split('=')[0]
        col_val = col.split('=')[1]
        out_col[col_name] = col_val
    return out_col

def get_snotel_info():
    #This is the latest CUAHSI API endpoint
    wsdlurl = 'https://hydroportal.cuahsi.org/Snotel/cuahsi_1_1.asmx?WSDL'
    sites = ulmo.cuahsi.wof.get_sites(wsdlurl)
    df_site_info = pd.DataFrame.from_dict(sites, orient='index').dropna()
    df_site_info['geometry'] = [Point(float(loc['longitude']), float(loc['latitude'])) for loc in df_site_info['location']]
    
    df_site_info['lat'] = df_site_info['location'].apply(pd.Series)['latitude']
    df_site_info['lon'] = df_site_info['location'].apply(pd.Series)['longitude']

    df_site_info['county'] = df_site_info['site_property'].apply(pd.Series)['county']
    df_site_info['state'] = df_site_info['site_property'].apply(pd.Series)['state']
    df_site_info['pos_accuracy_m'] = df_site_info['site_property'].apply(pd.Series)['pos_accuracy_m']
    df_site_info['site_comments'] = df_site_info['site_property'].apply(pd.Series)['site_comments']
    

    df_site_comments = df_site_info['site_comments'].apply(snotel_site_comments).apply(pd.Series)
    for col in df_site_comments.columns:
        df_site_info[col] = df_site_comments[col]

    df_site_info['isActive'] = df_site_info['isActive'].map({'True': True, 'False': False})

    df_site_info = df_site_info.drop(columns=['location','site_property','site_comments'])
    df_site_info = df_site_info.astype({'beginDate': 'datetime64[ns]', 
                                        'endDate': 'datetime64[ns]',
                                        'elevation_m': float,
                                        'lat': float,
                                        'lon': float,
                                        'TimeZone': float,
                                        'isActive': bool})
    df_site_info = df_site_info.rename(columns={'HUC': 'HUC12'})
    df_site_info['HUC8'] = df_site_info['HUC12'].apply(snotel_huc12_to_huc8)
    
    df_site_info['endDate'] = df_site_info['endDate'].where(df_site_info['isActive']==False, datetime.datetime.today())
    df_site_info['delta_years'] = (df_site_info['endDate'] - df_site_info['beginDate']).dt.days/365.25

    gdf_site_info = gpd.GeoDataFrame(df_site_info, crs='EPSG:4326')

    return gdf_site_info

def get_snotel_vars_info(sitecode):
    # SWE
    # SNOTEL:WTEQ_H = SWE, Hourly, inches
    # SNOTEL:WTEQ_D = SWE, Daily, inches
    # SNOTEL:WTEQ_m = SWE, Monthly, inches
    # SNOTEL:WTEQ_sm = SWE, Semi-Monthly(?), inches

    # SNWD (Snow Depth)
    # SNOTEL:SNWD_H = Snow Depth, Hourly, inches
    # SNOTEL:SNWD_D = Snow Depth, Daily, inches
    # SNOTEL:SNWD_m = Snow Depth, Monthly, inches
    # SNOTEL:SNWD_sm = Snow Depth, Semi-Monthly(?), inches

    # PREC (cumulative precipitation)
    # SNOTEL:PRCP_m = Precipitation, Monthly, inches
    # SNOTEL:PRCP_sm = Precipitation, Semi-Monthly(?), inches
    # SNOTEL:PRCP_wy = Precipitation, Common Year 365 days, inches

    # PRCP (incremental precipitation)
    # SNOTEL:PRCP_m = Precipitation, Monthly, inches
    # SNOTEL:PRCP_sm = Precipitation, Semi-Monthly(?), inches
    # SNOTEL:PRCP_wy = Precipitation, Common Year 365 days, inches
    # SNOTEL:PRCP_y = Precipitation, Yearly, inches

    # PRCPSA (precipitation snow adjusted)
    # SNOTEL:PRCPSA_D = Precipitation, Daily, inches
    # SNOTEL:PRCPSA_m = Precipitation, Monthly, inches
    # SNOTEL:PRCPSA_sm = Precipitation, Semi-Monthly(?), inches
    # SNOTEL:PRCPSA_wy = Precipitation, Common Year 365 days, inches
    # SNOTEL:PRCPSA_y = Precipitation, Yearly, inches

    # TOBS
    # SNOTEL:TOBS_H = Temperature, Hourly, Fahrenheit
    # SNOTEL:TOBS_D = Temperature, Daily, Fahrenheit
    # SNOTEL:TOBS_m = Temperature, Monthly, Fahrenheit
    # SNOTEL:TOBS_sm = Temperature, Semi-Monthly(?), Fahrenheit

    # TMIN
    # SNOTEL:TMIN_D = Temperature, Daily, Fahrenheit
    # SNOTEL:TMIN_m = Temperature, Monthly, Fahrenheit
    # SNOTEL:TMIN_sm = Temperature, Semi-Monthly(?), Fahrenheit
    # SNOTEL:TMIN_wy = Temperature, Common Year 365 days, Fahrenheit
    # SNOTEL:TMIN_y = Temperature, Yearly, Fahrenheit

    # TMAX
    # SNOTEL:TMAX_D = Temperature, Daily, Fahrenheit
    # SNOTEL:TMAX_m = Temperature, Monthly, Fahrenheit
    # SNOTEL:TMAX_sm = Temperature, Semi-Monthly(?), Fahrenheit
    # SNOTEL:TMAX_wy = Temperature, Common Year 365 days, Fahrenheit
    # SNOTEL:TMAX_y = Temperature, Yearly, Fahrenheit

    # TAVG
    # SNOTEL:TAVG_D = Temperature, Daily, Fahrenheit
    # SNOTEL:TAVG_m = Temperature, Monthly, Fahrenheit
    # SNOTEL:TAVG_sm = Temperature, Semi-Monthly(?), Fahrenheit
    # SNOTEL:TAVG_wy = Temperature, Common Year 365 days, Fahrenheit
    # SNOTEL:TAVG_y = Temperature, Yearly, Fahrenheit

    wsdlurl = 'https://hydroportal.cuahsi.org/Snotel/cuahsi_1_1.asmx?WSDL'
    variables_info = ulmo.cuahsi.wof.get_site_info(wsdlurl, sitecode)
    variables_dict = {}
    for i in variables_info['series'].keys():
        #print(i, variables_info['series'][i]['variable']['name'])
        variables_dict[i] = variables_info['series'][i]['variable']['name']
    return variables_dict

def get_snotel_data(sitecode, variablecode, start_date='1950-10-01',end_date= datetime.date.today()):
    print(sitecode,variablecode)
    # source: https://snowex-2021.hackweek.io/tutorials/geospatial/SNOTEL_query.html
    #print(sitecode, variablecode, start_date, end_date)


    wsdlurl = 'https://hydroportal.cuahsi.org/Snotel/cuahsi_1_1.asmx?WSDL'

    # Get Variable Information
    var_info = ulmo.cuahsi.wof.get_variable_info(wsdlurl, variablecode)
    units = var_info['units']['abbreviation']
    temporal_resolution = var_info['time']['units']['abbreviation']


    output_dict = None
    try:
        #Request data from the server
        site_values = ulmo.cuahsi.wof.get_values(wsdlurl, sitecode, variablecode, start=start_date, end=end_date)
        #Convert to a Pandas DataFrame   
        values_df = pd.DataFrame.from_dict(site_values['values'])
        #Parse the datetime values to Pandas Timestamp objects
        values_df['datetime'] = pd.to_datetime(values_df['datetime'], utc=True)
        #Set the DataFrame index to the Timestamps
        values_df = values_df.set_index('datetime')
        #Convert values to float and replace -9999 nodata values with NaN
        values_df['value'] = pd.to_numeric(values_df['value']).replace(-9999, np.nan)
        #Remove any records flagged with lower quality
        values_df = values_df[values_df['quality_control_level_code'] == '1']
        output_dict = {'values':values_df,
                    'units':units,
                    'temporal_resolution':temporal_resolution}
        if output_dict['units'] == 'in':
            output_dict['values']['value'] = output_dict['values']['value']*25.4 # in to mm
        if output_dict['units'] == 'degF':
            output_dict['values']['value'] = (output_dict['values']['value']-32.0)*(5.0/9.0) # degF to degC

    except:
        print("Unable to fetch %s" % variablecode)


    return output_dict