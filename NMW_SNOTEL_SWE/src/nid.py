import os# EWN Data Dir
import socket

# Data Directory
hostname = socket.gethostname()
if hostname == 'en4201928-ssebe.3500.dhcp.asu.edu':
    ewn_data_dir = '/Users/amoiz2/Servers/phx/data'
else:
    ewn_data_dir = '/data/EWN/amoiz/data'
nwm_data_dir = os.path.join(ewn_data_dir,'nwm3')

#----------------------------------NID (USACE)------------------------------
nid_dir = os.path.join(ewn_data_dir,'nid')
nid_national_files = {'gpkg_national_nid':os.path.join(nid_dir,'nation.gpkg'),
                      'csv_national_nid':os.path.join(nid_dir,'nation.csv'),}
