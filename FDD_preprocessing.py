# -*- coding: utf-8 -*-
# 3S Solar Plus - Master Project 
# Author: Hugo Quest (hugo.quest@hotmail.com)

import numpy as np
import pandas as pd
import datetime as dt
import math
import ipywidgets as widgets
from IPython.display import clear_output
import glob

import matplotlib.pyplot as plt

import warnings
warnings.simplefilter('ignore')

######################

def Get_data(name):
    
    DATA_FOLDER = 'data/'
    
    if(name == 'Bike stand'):
        
        SUB_FOLDER = '01_Old-MSII-80cells_Patricks-bicycle-stand/'

        #dataIV = pd.read_csv(DATA_FOLDER + SUB_FOLDER + 'thinkee daten - IV - July-2020 to January-2021 - 2.csv', sep=';')
        dataIV = pd.read_excel(DATA_FOLDER + SUB_FOLDER + 'thinkee daten - IV - July-2020 to January-2021 - 2.xls', sheet_name='Sheet1')

        ### Create two data sets for current and voltage
        dataI = dataIV[dataIV['units']=='A'].reset_index(drop=True)
        dataV = dataIV[dataIV['units']=='V']

        dataI.rename(columns = {'value':'Current'}, inplace = True)
        dataV.rename(columns = {'value':'Voltage'}, inplace = True) 

        dataV = dataV.drop(['val_id', 'units', 'commandId'], axis=1)
        dataI = dataI.drop(['val_id', 'units', 'commandId'], axis=1)

        ### Convert strings to datetime format
        dataI['time'] =  dataI['time'].apply(lambda x: dt.datetime.strptime(x[:-5], '%Y-%m-%dT%H:%M:%S'))
        dataV['time'] =  dataV['time'].apply(lambda x: dt.datetime.strptime(x[:-5], '%Y-%m-%dT%H:%M:%S'))

        ### Merge the I and V data into one dataframe 
        data = dataV.merge(dataI, how='outer')
        data.rename(columns = {'time':'Time'}, inplace = True) 
    
    elif(name == 'Bernapark'):
        
        SUB_FOLDER = '05_Bernapark/MS_shaded'
        path = DATA_FOLDER + SUB_FOLDER
        all_files = glob.glob(path + "/*.csv")

        li = []

        for filename in all_files:
            df = pd.read_csv(filename, index_col=None, header=0)
            li.append(df)

        data = pd.concat(li, axis=0, ignore_index=True)

        ### Cleaning
        data['Time'] =  data['Time'].apply(lambda x: dt.datetime.strptime(x, '%d.%m.%Y %H:%M'))
        data.rename(columns = {'P 1.1.9 I (A)':'Current', 'P 1.1.9 V (V)':'Voltage', 'PB 1.1.9 V (V)':'Optimised_Voltage'}, inplace = True) 
        data = data.drop('Optimised_Voltage', axis=1)
        
    elif(name == 'Linder'):
        
        SUB_FOLDER = '12_Schorenstrasse/DC_data/'
        path = DATA_FOLDER + SUB_FOLDER
        all_files = glob.glob(path + "/*.csv")

        li = []

        for filename in all_files:
            df = pd.read_csv(filename, index_col=None, header=0)
            li.append(df)

        data = pd.concat(li, axis=0, ignore_index=True)

        ### Cleaning
        data['Time'] =  data['Time'].apply(lambda x: dt.datetime.strptime(x, '%d.%m.%Y %H:%M'))
        data.rename(columns = {'P 1.2.11 I (A)':'Current', 'P 1.2.11 V (V)':'Voltage'}, inplace = True) 
        data = data.sort_values(by='Time', ascending=True)
        data = data.reset_index(drop=True)
        data['Power'] = data['Current']*data['Voltage']
        
    elif(name == 'Linder_unshaded'):
        
        SUB_FOLDER = '12_Schorenstrasse/DC_data_unshaded/'
        path = DATA_FOLDER + SUB_FOLDER
        all_files = glob.glob(path + "/*.csv")

        li = []

        for filename in all_files:
            df = pd.read_csv(filename, index_col=None, header=0)
            li.append(df)

        data = pd.concat(li, axis=0, ignore_index=True)

        ### Cleaning
        data['Time'] =  data['Time'].apply(lambda x: dt.datetime.strptime(x, '%d.%m.%Y %H:%M'))
        data.rename(columns = {'P 1.1.6 I (A)':'Current', 'P 1.1.6 V (V)':'Voltage'}, inplace = True) 
        data = data.sort_values(by='Time', ascending=True)
        data = data.reset_index(drop=True)
        data['Power'] = data['Current']*data['Voltage']
    
    elif(name == 'Linder_New'):
        
        SUB_FOLDER = '12_Schorenstrasse/DC_data_new/'
        path = DATA_FOLDER + SUB_FOLDER
        all_files = glob.glob(path + "/*.csv")

        li = []

        for filename in all_files:
            df = pd.read_csv(filename, index_col=None, header=0)
            li.append(df)

        data = pd.concat(li, axis=0, ignore_index=True)

        ### Cleaning
        data['Time'] =  data['Time'].apply(lambda x: dt.datetime.strptime(x, '%d.%m.%Y %H:%M'))
        data.rename(columns = {'P 1.2.11 I (A)':'Current', 'P 1.2.11 V (V)':'Voltage'}, inplace = True) 
        data = data.sort_values(by='Time', ascending=True)
        data = data.reset_index(drop=True)
        data['Power'] = data['Current']*data['Voltage']
     
    elif(name == 'Linder_New_Unshaded'):
        
        SUB_FOLDER = '12_Schorenstrasse/DC_data_new_unshaded/'
        path = DATA_FOLDER + SUB_FOLDER
        all_files = glob.glob(path + "/*.csv")

        li = []

        for filename in all_files:
            df = pd.read_csv(filename, index_col=None, header=0)
            li.append(df)

        data = pd.concat(li, axis=0, ignore_index=True)

        ### Cleaning
        data['Time'] =  data['Time'].apply(lambda x: dt.datetime.strptime(x, '%d.%m.%Y %H:%M'))
        data.rename(columns = {'P 1.1.6 I (A)':'Current', 'P 1.1.6 V (V)':'Voltage'}, inplace = True) 
        data = data.sort_values(by='Time', ascending=True)
        data = data.reset_index(drop=True)
        data['Power'] = data['Current']*data['Voltage']
    
    elif(name == 'Bernapark_Patina_Green'):
     
        SUB_FOLDER = '05_Bernapark/MSII_Patina_Green'
        SUB_FOLDER1 = '05_Bernapark/MSII_ModuleTemp'

        path = DATA_FOLDER + SUB_FOLDER
        path1 = DATA_FOLDER + SUB_FOLDER1
        all_files = glob.glob(path + "/*.csv")
        all_files1 = glob.glob(path1 + "/*.csv")

        li = []
        li1 = []

        for filename in all_files:
            df = pd.read_csv(filename, index_col=None, header=0)
            li.append(df)
        data = pd.concat(li, axis=0, ignore_index=True)

        for filename in all_files1:
            df = pd.read_csv(filename, sep=';', index_col=None, header=0)
            li1.append(df)
        data_moduleT = pd.concat(li1, axis=0, ignore_index=True)

        ### Cleaning
        data['Time'] =  data['Time'].apply(lambda x: dt.datetime.strptime(x, '%d.%m.%Y %H:%M'))
        data.rename(columns = {'P 1.2.12 I (A)':'Current', 'P 1.2.12 V (V)':'Voltage'}, inplace = True) 
        data['Power'] = data['Current']*data['Voltage']

        ### fixing the moduleT outputs 
        data_moduleT = data_moduleT.groupby('Timestamp').agg(sum).reset_index()
        data_moduleT['Timestamp'] =  data_moduleT['Timestamp'].apply(lambda x: dt.datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))
        data_moduleT.rename(columns = {'Timestamp':'Time', '4B - J55 - Patina Green':'Tmodule'}, inplace = True) 
        data_moduleT = data_moduleT[['Time', 'Tmodule']]
        data_moduleT = data_moduleT.set_index('Time')
        data_moduleT = data_moduleT.resample('15min').mean()
        data = data.merge(data_moduleT, left_on=data['Time'], right_on=data_moduleT.index)
        data = data.drop('key_0', axis=1)
    
    elif(name == 'Bernapark_Black'):
     
        SUB_FOLDER = '05_Bernapark/MSII_Black_New'
        SUB_FOLDER1 = '05_Bernapark/MSII_ModuleTemp'

        path = DATA_FOLDER + SUB_FOLDER
        path1 = DATA_FOLDER + SUB_FOLDER1
        all_files = glob.glob(path + "/*.csv")
        all_files1 = glob.glob(path1 + "/*.csv")

        li = []
        li1 = []

        for filename in all_files:
            df = pd.read_csv(filename, index_col=None, header=0)
            li.append(df)
        data = pd.concat(li, axis=0, ignore_index=True)

        for filename in all_files1:
            df = pd.read_csv(filename, sep=';', index_col=None, header=0)
            li1.append(df)
        data_moduleT = pd.concat(li1, axis=0, ignore_index=True)

        ### Cleaning
        data['Time'] =  data['Time'].apply(lambda x: dt.datetime.strptime(x, '%d.%m.%Y %H:%M'))
        data.rename(columns = {'P 1.2.18 I (A)':'Current', 'P 1.2.18 V (V)':'Voltage'}, inplace = True) 
        data['Power'] = data['Current']*data['Voltage']

        ### fixing the moduleT outputs 
        data_moduleT = data_moduleT.groupby('Timestamp').agg(sum).reset_index()
        data_moduleT['Timestamp'] =  data_moduleT['Timestamp'].apply(lambda x: dt.datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))
        data_moduleT.rename(columns = {'Timestamp':'Time', 'MS II Standard':'Tmodule'}, inplace = True) 
        data_moduleT = data_moduleT[['Time', 'Tmodule']]
        data_moduleT = data_moduleT.set_index('Time')
        data_moduleT = data_moduleT.resample('15min').mean()
        data = data.merge(data_moduleT, left_on=data['Time'], right_on=data_moduleT.index)
        data = data.drop('key_0', axis=1)
    
    
    print('\033[1m\033[92m >>> Data available for ' + str(name) + '\033[0m')
    return data

######################

def Get_data_solarlog(ID, inverter, string, year):
    
    DATA_FOLDER = 'data/'   
    SUB_FOLDER = 'Plant_data/'

    df = pd.read_csv(DATA_FOLDER + SUB_FOLDER + ID + '.txt', delimiter=';')
    df = df[df['yield']!='yield']
    df.iloc[:,1:] = df.iloc[:,1:].astype(float)

    Ninverters = df['inverter id'].unique()
    Nstrings = df['mpp count'][0]
    
    if(inverter in Ninverters):

        data = {name: pd.DataFrame() for name in Ninverters}

        for inv in Ninverters:
            data[inv] = df[df['inverter id']==inv]
            data[inv].rename(columns = {'datetime':'Time'}, inplace = True) 
            data[inv]['Time'] =  data[inv]['Time'].apply(lambda x: dt.datetime.strptime(x, '%d.%m.%y %H:%M:%S'))
            for s in range(int(Nstrings)):
                data[inv]['udc(mpp{0:d})'.format(s+1)] = data[inv]['udc(mpp{0:d})'.format(s+1)].replace(0, int(0))
                data[inv]['pdc(mpp{0:d})'.format(s+1)] = data[inv]['pdc(mpp{0:d})'.format(s+1)].replace(0, int(0))
                data[inv]['idc(mpp{0:d})'.format(s+1)] = data[inv]['pdc(mpp{0:d})'.format(s+1)]/data[inv]['udc(mpp{0:d})'.format(s+1)]
                data[inv]['idc(mpp{0:d})'.format(s+1)] = data[inv]['idc(mpp{0:d})'.format(s+1)].replace(np.nan, 0)
        data = data[inverter]
        
        if(string <= Nstrings):
            data = data[['Time', 'pdc(mpp{0:d})'.format(string), 'udc(mpp{0:d})'.format(string), 'idc(mpp{0:d})'.format(string)]]
            data.rename(columns = {'idc(mpp{0:d})'.format(string):'Current', 'udc(mpp{0:d})'.format(string):'Voltage', 'pdc(mpp{0:d})'.format(string):'Power'}, inplace = True) 
            data.reset_index(drop=True,inplace=True)
            
            data['year'] = data['Time'].apply(lambda x: x.year)
            Years = data['year'].unique()
            if(year in Years):
                data = data[data['year']==year]       
                print('\033[1m\033[92m >>> Data available for Inverter ' + str(int(inverter)) + ', string ' + str(int(string)) + ', in ' + str(int(year)) + '\033[0m \n')
                return data
            else:
                print('\033[1m\033[91m !!! Year not available, choose between ' + str(int(Years[0])) + ' and ' + str(int(Years[len(Years)-1])) + '\033[0m \n')
            
        else:
            print('\033[1m\033[91m !!! String not available, choose between 1 and ' + str(int(Nstrings)) + '\033[0m \n')
    else:
        print('\033[1m\033[91m !!! Inverter not available, choose between 1 and ' + str(int(len(Ninverters))) + '\033[0m \n')

######################

def Get_data_solarlog_json(ID, inverter, string, year):
    
    DATA_FOLDER = 'data/'   
    SUB_FOLDER = 'Plant_data_json/'
    path = DATA_FOLDER + SUB_FOLDER + str(ID)

    all_files = glob.glob(path + "/*.csv")
    Ninverters = len(all_files)
    inv_ID = []
    for i in range(Ninverters):
        inv_ID.append(i+1)
    print('\033[1m\033[92m >>> Plant ID ' + str(ID) + ', ' + str(Ninverters) + ' inverter(s) available' + '\n' + '\033[0m')

    if(inverter not in inv_ID):
        if(Ninverters==0):
            print('\033[1m\033[91m !!! No inverters available for this plant' + '\033[0m \n')
        else:
            print('\033[1m\033[91m !!! Inverter not available, choose between 1 and ' + str(Ninverters) + '\033[0m \n')
    else:
        data = {name: pd.DataFrame() for name in inv_ID}  
        count=1
        for filename in all_files:
            data[count] = pd.read_csv(filename)
            data[count].rename(columns={'Unnamed: 0': 'Time'}, inplace=True)
            data[count]['Time'] = pd.to_datetime(data[count]['Time'])
            count+=1    
        data = data[inverter]
        Nstrings = sum(data.columns.str.contains('Pdc'))
        if(string <= Nstrings):
            data = data[['Time', 'Pdc{0:d}'.format(string), 'Udc{0:d}'.format(string)]]
            data.rename(columns={'Pdc{0:d}'.format(string):'Power', 'Udc{0:d}'.format(string):'Voltage'}, inplace=True)
            data['Current'] = data['Power']/data['Voltage']
            data['Current'] = data['Current'].replace(np.nan,0)
            data['Power'] = data['Power'].astype(float)
            data['Voltage'] = data['Voltage'].astype(float)
            data['Current'] = data['Current'].astype(float)
            Years = data['Time'].dt.year.unique()
            #data = data[data['Power'] < data['Power'].mean()+5*data['Power'].std()]
            #data = data[data['Power'] > 100]
            #data = data[data['Voltage'] < data['Voltage'].mean()+5*data['Voltage'].std()]
            #data = data[data['Voltage'] > -1]
            #data = data[data['Current'] < data['Current'].mean()+5*data['Current'].std()]
            #data = data[data['Current'] > -1]
            if(year in Years):
                data = data[data['Time'].dt.year == year].reset_index(drop=True)
                if(len(data['Power'].unique())==1):
                    print('\033[1m\033[91m !!! All values are 0 for selected parameters ' + '\033[0m \n')
                    return data
                else:
                    print('\033[1m\033[92m >>> Data available for Inverter ' + str(inverter) + ', string ' + str(string) + '/' + str(Nstrings) + ', in ' + str(year) + '\033[0m \n')
                    return data
            elif(year=='all'):
                if(len(data['Power'].unique())==1):
                    print('\033[1m\033[91m !!! All values are 0 for selected parameters ' + '\033[0m \n')
                    return data
                else:
                    print('\033[1m\033[92m >>> Data available for Inverter ' + str(inverter) + ', string ' + str(string) + '/' + str(Nstrings) + ', all years' + '\033[0m \n')
                    return data
            else:
                print('\033[1m\033[91m !!! Year not available, choose between ' + str(Years[0]) + ' and ' + str(Years[len(Years)-1]) + '\033[0m \n')
        elif(Nstrings==0):
            print('\033[1m\033[91m !!! No strings available' + '\033[0m \n')
        else:
            print('\033[1m\033[91m !!! String not available, choose between 1 and ' + str(Nstrings) + '\033[0m \n')


###################### 

def Get_data_solarlog_json_other(ID, inverter, string, year):
    
    DATA_FOLDER = 'data/'   
    SUB_FOLDER = 'Plant_data_json_other/'
    path = DATA_FOLDER + SUB_FOLDER + str(ID)

    all_files = glob.glob(path + "/*.csv")
    Ninverters = len(all_files)
    inv_ID = []
    for i in range(Ninverters):
        inv_ID.append(i+1)
    print('\033[1m\033[92m >>> Plant ID ' + str(ID) + ', ' + str(Ninverters) + ' inverter(s) available' + '\n' + '\033[0m')

    if(inverter not in inv_ID):
        if(Ninverters==0):
            print('\033[1m\033[91m !!! No inverters available for this plant' + '\033[0m \n')
        else:
            print('\033[1m\033[91m !!! Inverter not available, choose between 1 and ' + str(Ninverters) + '\033[0m \n')
    else:
        data = {name: pd.DataFrame() for name in inv_ID}  
        count=1
        for filename in all_files:
            data[count] = pd.read_csv(filename)
            data[count].rename(columns={'Unnamed: 0': 'Time'}, inplace=True)
            data[count]['Time'] = pd.to_datetime(data[count]['Time'])
            count+=1    
        data = data[inverter]
        Nstrings = sum(data.columns.str.contains('Pdc'))
        if(string <= Nstrings):
            data = data[['Time', 'Pdc{0:d}'.format(string), 'Udc{0:d}'.format(string)]]
            data.rename(columns={'Pdc{0:d}'.format(string):'Power', 'Udc{0:d}'.format(string):'Voltage'}, inplace=True)
            data['Current'] = data['Power']/data['Voltage']
            data['Current'] = data['Current'].replace(np.nan,0)
            data['Power'] = data['Power'].astype(float)
            data['Voltage'] = data['Voltage'].astype(float)
            data['Current'] = data['Current'].astype(float)
            Years = data['Time'].dt.year.unique()
            #data = data[data['Power'] < data['Power'].mean()+5*data['Power'].std()]
            #data = data[data['Power'] > 100]
            #data = data[data['Voltage'] < data['Voltage'].mean()+5*data['Voltage'].std()]
            #data = data[data['Voltage'] > -1]
            #data = data[data['Current'] < data['Current'].mean()+5*data['Current'].std()]
            #data = data[data['Current'] > -1]
            if(year in Years):
                data = data[data['Time'].dt.year == year].reset_index(drop=True)
                if(len(data['Power'].unique())==1):
                    print('\033[1m\033[91m !!! All values are 0 for selected parameters ' + '\033[0m \n')
                    return data
                else:
                    print('\033[1m\033[92m >>> Data available for Inverter ' + str(inverter) + ', string ' + str(string) + '/' + str(Nstrings) + ', in ' + str(year) + '\033[0m \n')
                    return data
            elif(year=='all'):
                if(len(data['Power'].unique())==1):
                    print('\033[1m\033[91m !!! All values are 0 for selected parameters ' + '\033[0m \n')
                    return data
                else:
                    print('\033[1m\033[92m >>> Data available for Inverter ' + str(inverter) + ', string ' + str(string) + '/' + str(Nstrings) + ', all years' + '\033[0m \n')
                    return data
            else:
                print('\033[1m\033[91m !!! Year not available, choose between ' + str(Years[0]) + ' and ' + str(Years[len(Years)-1]) + '\033[0m \n')
        elif(Nstrings==0):
            print('\033[1m\033[91m !!! No strings available' + '\033[0m \n')
        else:
            print('\033[1m\033[91m !!! String not available, choose between 1 and ' + str(Nstrings) + '\033[0m \n')

######################             
            
def Widgets_other():

   # Widget to choose data source
    b_data = widgets.Dropdown(
        options=['Bike stand', 'Bernapark', 'Linder', 'Linder_unshaded', 'Linder_New', 'Linder_New_Unshaded',
                 'Bernapark_Black', 'Bernapark_Patina_Green'],
        description='Data source',
        disabled=False,
    )

    b = widgets.interactive(Get_data, {'manual': True}, name = b_data)
    return b

######################  

def Widgets_solarlog_csv():

   # Widget to choose plant ID
    b_ID = widgets.Dropdown(
        options=['808144799'],
        description='Plant ID',
        disabled=False,
    )

    # Widget to choose inverter
    b_inv = widgets.Dropdown(
        options=[1, 2, 3, 4, 5],
        value=1,
        description='Inverter',
        disabled=False,
    )

    # Widget to choose string
    b_str = widgets.Dropdown(
        options=[1, 2, 3, 4],
        value=1,
        description='String',
        disabled=False,
    )

    # Widget to choose string
    b_year = widgets.Dropdown(
        options=[2010,2011,2012,2013,2014,2015,\
                 2016,2017,2018,2019,2020,2021],
        value=dt.datetime.now().year,
        description='Year',
        disabled=False,
    )

    b = widgets.interactive(Get_data_solarlog , {'manual': True}, ID = b_ID, inverter = b_inv, string = b_str, year = b_year)
    return b

######################  

def Widgets_solarlog_json():

    folder = 'data/14_SolarLog/API_data.xlsx'
    df_URL2 = pd.read_excel(folder, sheet_name='Sheet2')
    ID_list = df_URL2['ID'].to_list()

    # Widget to choose plant ID
    b_ID = widgets.Dropdown(
        options=ID_list,
        description='Plant ID',
        disabled=False,
    )

    # Widget to choose inverter
    b_inv = widgets.Dropdown(
        options=[1, 2, 3, 4, 5, 6, 7, 8, 9 ,10],
        value=1,
        description='Inverter',
        disabled=False,
    )

    # Widget to choose string
    b_str = widgets.Dropdown(
        options=[1, 2, 3, 4],
        value=1,
        description='String',
        disabled=False,
    )

    # Widget to choose string
    b_year = widgets.Dropdown(
        options=['all',2010,2011,2012,2013,2014,2015,\
                 2016,2017,2018,2019,2020,2021],
        value=dt.datetime.now().year,
        description='Year',
        disabled=False,
    )

    b = widgets.interactive(Get_data_solarlog_json , {'manual': True}, ID = b_ID, inverter = b_inv, string = b_str, year = b_year)
    return b

######################  

def Widgets_solarlog_json_other():

    folder = 'data/14_SolarLog/API_data.xlsx'
    df_URL2 = pd.read_excel(folder, sheet_name='Long_term')
    ID_list = df_URL2['ID'].to_list()

    # Widget to choose plant ID
    b_ID = widgets.Dropdown(
        options=ID_list,
        description='Plant ID',
        disabled=False,
    )

    # Widget to choose inverter
    b_inv = widgets.Dropdown(
        options=[1, 2, 3, 4, 5, 6, 7, 8, 9 ,10],
        value=1,
        description='Inverter',
        disabled=False,
    )

    # Widget to choose string
    b_str = widgets.Dropdown(
        options=[1, 2, 3, 4],
        value=1,
        description='String',
        disabled=False,
    )

    # Widget to choose string
    b_year = widgets.Dropdown(
        options=['all',2010,2011,2012,2013,2014,2015,\
                 2016,2017,2018,2019,2020,2021],
        value=dt.datetime.now().year,
        description='Year',
        disabled=False,
    )

    b = widgets.interactive(Get_data_solarlog_json_other , {'manual': True}, ID = b_ID, inverter = b_inv, string = b_str, year = b_year)
    return b

######################  

def Filter_data(data, meta):

    data = data.set_index('Time')
    data.index = data.index.tz_localize('UTC', ambiguous = 'infer')

    data = data[(data.index.time>=dt.time(6)) & (data.index.time<=dt.time(21))]

    data = data[data['Power'] < meta['kWp'][0]*1000]
    data = data[data['Current'] < meta['kWp'][0]*1000]
    data = data[data['Power'] > 0]

    return data

######################  
                        
def Pre_process(data):
    
    # create a Power column (I*V)
    data['Power'] = data['Current']*data['Voltage']
    
    # add separate date, hour, month, year, month_year columns
    data['date'] = data['Time'].apply(lambda x: x.date())
    data['hour'] = data['Time'].apply(lambda x: x.time())
    data['month'] = data['Time'].apply(lambda x: x.month)
    data['year'] = data['Time'].apply(lambda x: x.year)
    data['month_year'] = data['Time'].dt.to_period('M')
    #data['xaxis'] = data['Time'].apply(lambda x: x.minute/60 + x.hour)

    return(data)

######################   
        
def Data_plot(data):
    COLUMNS = 1
    ROWS = 2
    col = ['Voltage', 'Current']
    y_ax = ['Voltage [V]', 'Current [A]']

    fig, axs = plt.subplots(ROWS, COLUMNS, figsize=(15, 7))

    for j in range(2):
    
        ax = axs[j]
    
        ax.scatter(data['Time'], data[col[j]] , marker='.', linestyle=':')
        ax.set_title(col[j])
        ax.set_ylabel(y_ax[j])
      
    plt.tight_layout()

######################    
    
def Monthly_plot(data, var, monthyear):
    
    plot_data = data[(data['month_year']==monthyear)]
    subplot_data = plot_data.groupby('date').agg(list).reset_index()
    
    if(var=='Current'):
        var_label = var + ' [A]'
    elif(var=='Voltage'):
        var_label = var + ' [V]'
    elif(var=='Power'):
        var_label = var + ' [W]'

    fig, ax = plt.subplots(figsize = (20,7))
    for i in range(len(subplot_data)):
        plt.plot(subplot_data.iloc[i,:]['Time'], subplot_data.iloc[i,:][var], marker='.', markersize=0.2, color='teal')
        plt.xlabel('Time')
        plt.ylabel(var_label)  

######################  

def Visualise_monthly_output(data, Months):
    
    # Widget to choose the period 
    b_date = widgets.Dropdown(
        options=Months,
        value=Months[0],
        description='Date',
        disabled=False,
    )

    # Widget to choose variable
    b_var = widgets.Dropdown(
        options=['Current', 'Voltage', 'Power'],
        value='Power',
        description='Variable',
        disabled=False,
    )

    # Widget to run function
    b_run = widgets.Button(description="Run")
    def on_button_clicked(b):
        clear_output(wait=True)
        Monthly_plot(data, b_var.value, b_date.value)
        display(widgets.HBox([b_run, b_date, b_var]))

    b_run.on_click(on_button_clicked)

    display(widgets.HBox([b_run, b_date, b_var]))        

######################

def Hourly_output_OLD(variable, data, Nmonths, Months, Hrange):
    
    # Hourly produced outputs per day
    df = []
    date_vec = []
    monthyear_vec = []
    plot_data = data.groupby('date').agg(lambda x: list(x))
    plot_data['month_year'] = plot_data['month_year'].apply(lambda x: x[0]) 

    for i in range(Nmonths):
        sub_data = plot_data[plot_data.month_year == Months[i]]

        for j in range(len(sub_data)):
            sub_sub_data = sub_data.iloc[j]
            date_vec.append(sub_sub_data.name)
            monthyear_vec.append(sub_sub_data['month_year'])
            interm = pd.DataFrame([sub_sub_data['hour'],sub_sub_data[variable]]).T
            interm.columns=['hour',variable]
            interm['hour'] = interm['hour'].apply(lambda x: x.hour)
            interm = interm.groupby('hour').agg(lambda x: x.mean()).reset_index()

            vec=np.zeros(len(Hrange))
            vec[vec==0] = np.nan

            for i in Hrange:
                if(i in interm['hour'].values):
                      vec[i-Hrange[0]] = interm[variable][interm['hour']==i]
            df.append(vec)

    df = pd.DataFrame(df, columns=Hrange)
    df['date'] = date_vec
    df['month_year'] = monthyear_vec
    
    return df

######################

def Resample_data(data):
    
    # Resample to hourly data
    df = data.resample('H', on = 'Time', base = 0).mean().reset_index()
    
    # Include all hours of the day by re-indexing
    start = df['Time'][0]
    end = df['Time'][len(df)-1]
    index = pd.date_range(dt.datetime(start.year, start.month, start.day, 0), dt.datetime(end.year, end.month, end.day, 23), freq='H')
    df = df.set_index('Time').reindex(index=index)
    df =  df.reset_index()
    df.rename(columns={'index':'Time'}, inplace=True)
    df['date'] = df['Time'].apply(lambda x: x.date())
    df['hour'] = df['Time'].apply(lambda x: x.time())
    df['month_year'] = df['Time'].dt.to_period('M')
    
    return df

######################

def Hourly_output(var, data, Hrange):
    
    df = Resample_data(data)
    df = df.groupby('date').agg(list).reset_index()
    
    # Output the wanted variable
    df_out = pd.DataFrame(df[var].tolist())
    df_out.index = df.date
    df_out = df_out[Hrange]
    df_out = df_out.reset_index()
    df_out['month_year'] = df['month_year'].apply(lambda x: x[0])
    
    return df_out

######################
    
def Max_output(df, Hrange):

    df_max = df.groupby('month_year').max().reset_index()
    df_max['std'] = df_max.apply(lambda x: np.std(x.loc[Hrange[0]:Hrange[len(Hrange)-1]]), axis=1)
    
    return df_max

######################

def Min_output(df, Hrange):

    df_min = df.groupby('month_year').min().reset_index()
    df_min['std'] = df_min.apply(lambda x: np.std(x.loc[Hrange[0]:Hrange[len(Hrange)-1]]), axis=1)
    
    return df_min

######################

def Plot_threshold(df, th, var, Nmonths, Hrange):
    
    COLUMNS = 3
    ROWS = math.ceil((Nmonths)/COLUMNS)
    
    if(var=='Current'):
        var_label = var + ' [A]'
    elif(var=='Voltage'):
        var_label = var + ' [V]'
    elif(var=='Power'):
        var_label = var + ' [W]'
    
    if(Nmonths>3):
        fig, axs = plt.subplots(ROWS, COLUMNS, figsize=(15, 12), sharey=True, sharex=True)
    else:
        fig, axs = plt.subplots(ROWS, COLUMNS, figsize=(15, 5), sharey=True, sharex=True)
        
    for i in range(Nmonths):
        
        if(Nmonths>3):
            current_column = i%COLUMNS
            current_row = i//COLUMNS
            ax = axs[current_row, current_column]
        else:
            ax = axs[i]

        y = df.loc[i,Hrange[0]:Hrange[len(Hrange)-1]].values.astype(float)
        x = Hrange
        std = df['std'].mean()
        ax.plot(x, y, marker='.', linestyle='-')
        ax.plot(x, y+th*std, linestyle=':', color='g')
        ax.plot(x, y-th*std, linestyle=':',color='g')
        ax.set_title(df.iloc[i]['month_year'])

    # Set labels
    if(Nmonths>3):
        plt.setp(axs[-1, :], xlabel='Time')
        plt.setp(axs[:, 0], ylabel=var_label)    
        plt.tight_layout()
    else:
        plt.setp(axs[-2], xlabel='Time')   
        plt.tight_layout()

######################    
    
def Visualise_thresholds(P_max, I_max, V_max, Nmonths, Hrange):

    # Widget to choose variable
    b_var = widgets.Dropdown(
        options=['Current', 'Voltage', 'Power'],
        value='Power',
        description='Variable',
        disabled=False,
    )

    # Widget to choose threshold
    b_th = widgets.FloatText(
        value=0.5,
        description='Threshold:',
        disabled=False
    )

    # Widget to run function
    b_run = widgets.Button(description="Run")
    def on_button_clicked(b):
        clear_output(wait=True)
        if(b_var.value=='Current'):
            var_max = I_max
        elif(b_var.value=='Voltage'):
            var_max = V_max
        elif(b_var.value=='Power'):
            var_max = P_max
        Plot_threshold(var_max, b_th.value, b_var.value, Nmonths, Hrange)
        display(widgets.HBox([b_run, b_var, b_th]))

    b_run.on_click(on_button_clicked)

    display(widgets.HBox([b_run, b_var, b_th]))
    
######################