import alpha_vantage
from alpha_vantage.timeseries import TimeSeries

ts = TimeSeries(key = 'PJSJKANB2G64XENP', output_format= 'pandas')

#get json object with intraday data and another with the call's metadata
data, meta_data = ts.get_intraday('GBPUSD', interval = '5min', outputsize = 'full')

print(data.head(2))

