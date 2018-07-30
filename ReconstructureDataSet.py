
from pandas import read_csv
from datetime import datetime
from pandas.tests.io.parser import index_col
from statsmodels.tsa.base.datetools import date_parser

data_dir = 'C:/Users/sampei/Desktop/kaggle/BeijingAirPollution/raw.csv'
mod_data_dir = 'C:/Users/sampei/Desktop/kaggle/BeijingAirPollution/pollution.csv'

#=====Restructureing data==========
def parse(x):
    return datetime.strptime(x, '%Y %m %d %H')
# parsing date column
dataset = read_csv(data_dir, parse_dates=[['year', 'month', 'day', 'hour']], index_col=0, date_parser=parse)

#drop No column
dataset.drop('No', axis=1, inplace=True)

#defining column
dataset.columns = ['pollution', 'dew', 'temp', 'press', 'wnd_dir', 'wnd_spd', 'snow', 'rain']
dataset.index.name = 'date'

#fiiling NAN of pollution by 0
dataset['pollution'].fillna(0, inplace=True)

#cutting the first 24 hours
dataset = dataset[24:]

#saving the data at the form of csv
dataset.to_csv(mod_data_dir)

