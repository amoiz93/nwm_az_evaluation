import pandas as pd
import geopandas as gpd
import glob
import os

def get_fluxnet_site_info(df_fluxnet_site_info):
    dfs_fluxnet_site_info = []
    for site in list(df_fluxnet_site_info['SITE_ID'].unique()):
        df_fluxnet_site_info_sel = df_fluxnet_site_info[df_fluxnet_site_info['SITE_ID']==site]
        df_fluxnet_site_info_sel = pd.DataFrame(df_fluxnet_site_info_sel['DATAVALUE'].values,index=df_fluxnet_site_info_sel['VARIABLE'].values,columns=[site]).T
        sel_cols = [
                    'SITE_NAME',
                    'LOCATION_LAT',
                    'LOCATION_LONG',
                    'LOCATION_ELEV',
                    'UTC_OFFSET',
                    # 'STATE',
                    'COUNTRY',
                    'UTC_OFFSET',
                    'FLUX_MEASUREMENTS_OPERATIONS',
                    'FLUX_MEASUREMENTS_METHOD',
                    'FLUX_MEASUREMENTS_DATE_START',
                    'FLUX_MEASUREMENTS_DATE_END',
                    'IGBP',
                    'IGBP_COMMENT',
                    'TERRAIN',
                    'ASPECT',
                    'WIND_DIRECTION',
                    'URL_AMERIFLUX'
                    ]
        sel_cols_fin = df_fluxnet_site_info_sel.columns.intersection(sel_cols)
        df_fluxnet_site_info_sel = df_fluxnet_site_info_sel[sel_cols_fin]
        df_fluxnet_site_info_sel = df_fluxnet_site_info_sel.loc[:,~df_fluxnet_site_info_sel.columns.duplicated(keep='first')]
        dfs_fluxnet_site_info.append(df_fluxnet_site_info_sel)
    dfs_fluxnet_site_info = pd.concat(dfs_fluxnet_site_info)
    dfs_fluxnet_site_info = dfs_fluxnet_site_info.rename(columns={'SITE_NAME':'name',
                                                                'LOCATION_LAT':'lat',
                                                                'LOCATION_LONG':'lon',
                                                                'LOCATION_ELEV':'elev',
                                                                'COUNTRY':'country',
                                                                'UTC_OFFSET':'utc_offset',
                                                                'FLUX_MEASUREMENTS_OPERATIONS':'FM_operations',
                                                                'FLUX_MEASUREMENTS_METHOD':'FM_method',
                                                                'FLUX_MEASUREMENTS_DATE_START':'FM_date_start',
                                                                'FLUX_MEASUREMENTS_DATE_END':'FM_date_end',
                                                                'IGBP':'igbp',
                                                                'IGBP_COMMENT':'igbp_comment',
                                                                'TERRAIN':'terrain',
                                                                'ASPECT':'aspect',
                                                                'WIND_DIRECTION':'wind_direction',
                                                                'URL_AMERIFLUX':'url'})
    gdf_fluxnet_site_info = gpd.GeoDataFrame(dfs_fluxnet_site_info,geometry=gpd.points_from_xy(dfs_fluxnet_site_info['lon'],dfs_fluxnet_site_info['lat']),crs='EPSG:4326')
    return gdf_fluxnet_site_info


def LE_to_ET(df_LE,df_TA,frequency):
    # df_LE: W m-2
    # df_TA: degC
    # df_ET: mm
    df_lambda_latent_heat = 2.501-(2.361*10**(-3))*df_TA # https://earthscience.stackexchange.com/questions/20733/fluxnet15-how-to-convert-latent-heat-flux-to-actual-evapotranspiration
    df_ET = df_LE/(df_lambda_latent_heat*10**(6))
    if frequency == 'hourly':
        df_ET = df_ET*(60*30) # kg/m2/s --> mm/30min
    elif frequency == 'daily':
        df_ET = df_ET*24*60*60 # kg/m2/s --> mm/day
    elif frequency == 'monthly':
        df_ET_days_in_month = df_ET.index.days_in_month
        df_ET = df_ET*(df_ET_days_in_month*24*60*60) # kg/m2/s --> mm/month
    return df_ET

def get_FLXUNET_data(sites,var,frequency,fluxnet_dir,gdf_fluxnet_sites):

    # 4) VARIABLE LIST (Selected)

    # Below is a list of the variable root names, descriptions, and units appearing in the FULLSET, SUBSET, and ERA5 files. Separate units are listed if different units are used in different temporal resolutions. See https://fluxnet.org/data/fluxnet2015-dataset/fullset-data-product/ and https://fluxnet.org/data/fluxnet2015-dataset/subset-data-product/ for a complete list of variable names and units in the FULLSET and SUBSET, respectively.

    # VARIABLE_ROOT	Description					Units
    # TA		Air temperature      				deg C
    # SW_IN_POT	Potential shortwave incoming radiation		W m−2
    # SW_IN 		Shortwave incoming radiation 			W m−2
    # LW_IN		Longwave incoming radiation 			W m−2
    # VPD		Vapor pressure saturation deficit		hPa
    # PA 		Atmospheric pressure 				kPa
    # P 		Precipitation 					mm (HH/HR) mm d−1 (DD-MM) mm y−1 (YY)
    # WS 		Wind speed 					m s −1
    # WD 		Wind direction 					Decimal degrees
    # RH 		Relative humidity			 	%
    # USTAR 		Friction velocity 				m s−1
    # NETRAD 		Net radiation 					W m−2
    # PPFD_IN		Incoming photosynthetic photon flux density	µmolPhoton m−2  s−1
    # PPFD_DIF	Diffuse PPFD_IN					µmolPhoton m−2  s−1
    # PPFD_OUT	Outgoing photosynthetic photon flux density	µmolPhoton m−2  s−1
    # SW_DIF		Diffuse SW_IN					W m−2
    # SW_OUT		Shortwave outgoing radiation			W m−2
    # LW_OUT		Longwave outgoing radiation			W m−2
    # CO2 		CO2 mole fraction 				µmolCO2  mol−1
    # TS 		Soil temperature 				deg C
    # SWC 		Soil water content 				%
    # G 		Soil heat flux 					W m−2
    # LE 		Latent heat flux 				W m−2
    # H 		Sensible heat flux 				W m−2
    # NEE 		Net Ecosystem Exchange 				µmolCO2  m−2  s−1 (HH/HR) gC m−2  d−1 (DD-MM) gC m−2 y−1 (YY)
    # RECO 		Ecosystem Respiration				µmolCO2  m−2  s−1 (HH/HR) gC m−2  d−1 (DD-MM) gC m−2 y−1 (YY)
    # GPP 		Gross Primary Production			µmolCO2  m−2  s−1 (HH/HR) gC m−2  d−1 (DD-MM) gC m−2 y−1 (YY)

    # A list of the most commonly seen qualifiers are provided here:

    # QUALIFIER	Description
    # _#		Layer qualifier, numeric index “#” increases with the depth, 1 is shallowest
    # _F		Gap-filled variable 
    # _QC		Quality flag; See USAGE NOTE for details
    # _NIGHT		Variable aggregated using only nighttime data
    # _DAY		Variable aggregated using only daytime data
    # _SD		Standard deviation
    # _SE		Standard Error
    # _MDS	 	Marginal Distribution Sampling gap-filling method
    # _ERA		Data filled by using ERA5 downscaling
    # _JSB		Longwave radiation calculated using the JSBACH algorithm (Sonke Zaehle)
    # _CORR		Energy fluxes corrected by energy balance closure correction factor (EBC_CF); See USAGE NOTE for details.
    # _CORR_25	Energy fluxes corrected by EBC_CF, 25th percentile; See USAGE NOTE for details.
    # _CORR_75	Energy fluxes corrected by EBC_CF, 75th percentile; See USAGE NOTE for details.
    # _METHOD		Method used to estimate uncertainty or energy balance closure correction
    # _RANDUNC	Random uncertainty
    # _CORR_JOINTUNC	Joint uncertainty combining from EBC_CF and random uncertainty
    # _VUT		Variable USTAR threshold for each year
    # _CUT		Constant USTAR threshold for each year
    # _REF		Most representative NEE after filtering using multiple USTAR thresholds; See USAGE NOTE for details.
    # _MEAN		Average NEE after filtering using multiple USTAR thresholds; See USAGE NOTE for details.
    # _USTAR50	NEE filtering by using the median value of the USTAR thresholds distribution; See USAGE NOTE for details.
    # _REF_JOINTUNC	Joint uncertainty combining from multiple USTAR thresholds and random uncertainty
    # _DT		Partitioning NEE using the daytime flux method, Lasslop et al. (2010)
    # _NT		Partitioning NEE using the nighttime flux method, Reichstein et al. (2005)
    # _SR		Partitioning NEE using the van Gorsel et al. (2009) method

    dfs_fluxnet_site_data_var = []
    for site in sites:
        fluxnet_site_data_dir = glob.glob(os.path.join(fluxnet_dir,f'AMF_{site}_FLUXNET*'))[0]
        utc_offset = gdf_fluxnet_sites.loc[site,'utc_offset']
        
        # Read Data
        if frequency == 'hourly':
            # Hourly Data
            fluxnet_site_data_csv = glob.glob(os.path.join(fluxnet_site_data_dir,f'AMF_{site}_FLUXNET_FULLSET_HH*.csv'))[0]
            df_fluxnet_site_data = pd.read_csv(fluxnet_site_data_csv)
            df_fluxnet_site_data.index = pd.to_datetime(df_fluxnet_site_data['TIMESTAMP_START'],format='%Y%m%d%H%M')
            df_fluxnet_site_data = df_fluxnet_site_data.where(df_fluxnet_site_data!=-9999.0)
            
        if frequency == 'daily':
            # Daily Data
            fluxnet_site_data_csv = glob.glob(os.path.join(fluxnet_site_data_dir,f'AMF_{site}_FLUXNET_FULLSET_DD*.csv'))[0]
            df_fluxnet_site_data = pd.read_csv(fluxnet_site_data_csv)
            df_fluxnet_site_data.index = pd.to_datetime(df_fluxnet_site_data['TIMESTAMP'],format='%Y%m%d')
            df_fluxnet_site_data = df_fluxnet_site_data.where(df_fluxnet_site_data!=-9999.0)

        if frequency == 'monthly':
            # Monthly Data
            fluxnet_site_data_csv = glob.glob(os.path.join(fluxnet_site_data_dir,f'AMF_{site}_FLUXNET_FULLSET_MM*.csv'))[0]
            df_fluxnet_site_data = pd.read_csv(fluxnet_site_data_csv)
            df_fluxnet_site_data.index = pd.to_datetime(df_fluxnet_site_data['TIMESTAMP'],format='%Y%m')
            df_fluxnet_site_data = df_fluxnet_site_data.where(df_fluxnet_site_data!=-9999.0)

        # Convert LE (W/m2) to ET (mm)
        if var == 'ET_F_MDS':
            df_fluxnet_site_data_var = LE_to_ET(df_LE=df_fluxnet_site_data['LE_F_MDS'],
                                                        df_TA=df_fluxnet_site_data['TA_F_MDS'],
                                                        frequency=frequency)
        elif var == 'ET_CORR':
            df_fluxnet_site_data_var = LE_to_ET(df_LE=df_fluxnet_site_data['LE_CORR'],
                                                        df_TA=df_fluxnet_site_data['TA_F_MDS'],
                                                        frequency=frequency)
        else:
            df_fluxnet_site_data_var = df_fluxnet_site_data[var]

        
        df_fluxnet_site_data_var.index.name = 'datetime'
        df_fluxnet_site_data_var.name = site
        df_fluxnet_site_data_var = df_fluxnet_site_data_var.to_frame()
        df_fluxnet_site_data_var.index = df_fluxnet_site_data_var.index - pd.Timedelta(hours=float(utc_offset))
        dfs_fluxnet_site_data_var.append(df_fluxnet_site_data_var)
    dfs_fluxnet_site_data_var = pd.concat(dfs_fluxnet_site_data_var,axis=1)
    return dfs_fluxnet_site_data_var
