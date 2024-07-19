from alpha_vantage.timeseries import TimeSeries
import os
import time

ts = TimeSeries(key = 'PJSJKANB2G64XENP', output_format= 'pandas')

try:
    data, meta_data = ts.get_intraday(symbol='TSLA', interval='60min', outputsize='full')
    print(data.head(50))
    print("Data received successfully!")
    time.sleep(15)
except ValueError as e:
    print(f"ValueError: {e}")
except Exception as e:
    print(f"An error occurred: {e}")

