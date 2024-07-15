import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mplfinance as mpf
import scipy
import math
import pandas_ta as ta
# from AVtest import data

# Constructs market profile using logarithmic pricing
# first_w varies weightage of prices depending on length of time from present
# atr_mult affects smoothing of KDE
# As prom_thresh >, only resistance levels with greater significance taken 

def find_levels( 
        price: np.array, atr: float,  # Log closing price, and log atr
        first_w: float = 0.1, 
        atr_mult: float = 1.0, 
        prom_thresh: float = 0.05
):

    # Setup weights
    last_w = 1.0
    w_step = (last_w - first_w) / len(price)
    weights = first_w + np.arange(len(price)) * w_step
    weights[weights < 0] = 0.0

    # Get kernel of price. 
    # bw_method = bandwidth of KDE
    kernal = scipy.stats.gaussian_kde(price, bw_method=atr*atr_mult, weights=weights)

    # Construct market profile
    # Step constructs grid with 200 equal spacings between max/min price
    min_v = np.min(price)
    max_v = np.max(price)
    step = (max_v - min_v) / 200
    price_range = np.arange(min_v, max_v, step)
    pdf = kernal(price_range)  # Market profile

    # Find significant peaks in the market profile
    pdf_max = np.max(pdf)
    prom_min = pdf_max * prom_thresh

    peaks, props = scipy.signal.find_peaks(pdf, prominence=prom_min)
    levels = [] 
    for peak in peaks:
        levels.append(np.exp(price_range[peak]))

    return levels, peaks, props, price_range, pdf, weights


def support_resistance_levels(
        data: pd.DataFrame, lookback: int, 
        first_w: float = 0.1, atr_mult:float=1.0, prom_thresh:float =0.05
):

    # Get log average true range, 
    atr = ta.atr(np.log(data['high']), np.log(data['low']), np.log(data['close']), lookback)

    all_levels = [None] * len(data)
    for i in range(lookback, len(data)):
        i_start  = i - lookback
        vals = np.log(data.iloc[i_start+1: i+1]['close'].to_numpy())
        levels, peaks, props, price_range, pdf, weights= find_levels(vals, atr.iloc[i], first_w, atr_mult, prom_thresh)
        all_levels[i] = levels
        
    return all_levels

def sr_penetration_signal(data: pd.DataFrame, levels: list):
    signal = np.zeros(len(data))
    curr_sig = 0.0
    close_arr = data['close'].to_numpy()
    for i in range(1, len(data)):
        if levels[i] is None:
            continue

        last_c = close_arr[i - 1]
        curr_c = close_arr[i]

        
        for level in levels[i]:
            if curr_c > level and last_c <= level: # Close cross above line
                curr_sig = 1.0
            elif curr_c < level and last_c >= level: # Close cross below line
                curr_sig = -1.0
        
        # If signal == 1, buy order / If signal == -1, sell order
        signal[i] = curr_sig
    return signal

def get_trades_from_signal(data: pd.DataFrame, signal: np.array):
    long_trades = []
    short_trades = []

    close_arr = data['close'].to_numpy()
    last_sig = 0.0
    open_trade = None
    idx = data.index
    for i in range(len(data)):
        # If previous signal is of the same type, breaks loop
        if signal[i] == 1.0 and last_sig != 1.0: 
            # If another trade is open, breaks loop
            if open_trade is not None:
                open_trade[2] = idx[i] 
                open_trade[3] = close_arr[i]
                short_trades.append(open_trade)

            open_trade = [idx[i], close_arr[i], -1, np.nan]
        if signal[i] == -1.0  and last_sig != -1.0: 
            if open_trade is not None:
                open_trade[2] = idx[i] 
                open_trade[3] = close_arr[i]
                long_trades.append(open_trade)

            open_trade = [idx[i], close_arr[i], -1, np.nan]

        last_sig = signal[i]

    # Object creation
    long_trades = pd.DataFrame(long_trades, columns=['entry_time', 'entry_price', 'exit_time', 'exit_price'])
    short_trades = pd.DataFrame(short_trades, columns=['entry_time', 'entry_price', 'exit_time', 'exit_price'])

    # Lot size provides magnitude to the profit calculated
    lot_size = float(1000)
    # Calculated trade profit based on percentage 
    long_trades['percent'] = (long_trades['exit_price'] - long_trades['entry_price']) / long_trades['entry_price'] 
    short_trades['percent'] = -1 * (short_trades['exit_price'] - short_trades['entry_price']) / short_trades['entry_price']
    # Places profit from each trade into an array
    long_trades['profit'] = long_trades['percent'] * lot_size
    short_trades['profit'] = short_trades['percent'] * lot_size
    long_trades = long_trades.set_index('entry_time')
    short_trades = short_trades.set_index('entry_time')
    return long_trades, short_trades 

# Stores currency data files
csv_storage = {
    "EURUSD": "D:\Downloads\EURUSD_Candlestick_1_D_BID_05.05.2003-28.10.2023.csv",
    "BTCUSDT": "D:\Downloads\BTCUSDT86400.csv",
    "TESLA": "D:\Downloads\esla-stock-price.csv",
}


if __name__ == '__main__':
   
    # Selects file from csv_storage
    file_choice = str(input("Select file: "))
    data = pd.read_csv(csv_storage.get(file_choice))
    # Type casts date column in csv file
    data['date'] = data['date'].astype('datetime64[s]')
    data = data.set_index('date')
    plt.style.use('dark_background') 
    levels = support_resistance_levels(data, 365, first_w=0.1, atr_mult=1.0)

    data['sr_signal'] = sr_penetration_signal(data, levels)
    data['log_ret'] = np.log(data['close']).diff().shift(-1)
    data['sr_return'] = data['sr_signal'] * data['log_ret']

    long_trades, short_trades = get_trades_from_signal(data, data['sr_signal'].to_numpy())

    long_profit = float(round(sum(long_trades['profit']),2))
    short_profit = float(round(sum(short_trades['profit']),2))

# Displays trade info for each currency data file
def test_trades():
    print("Long Trades")
    print (long_trades)
    print("")
    print("Short Trades")
    print (short_trades)
    print("")
    print("Trades taken: "+ str(len(long_trades)+len(short_trades)))
    print("Long Trades: $"+str(long_profit))
    print("Short Trades: $"+str(short_profit))
    print("Profit: $"+str(long_profit + short_profit))

# Data plotting function
def plotting():
    # Plot closing prices
    plt.figure(figsize=(20, 6))
    plt.plot(data.index, data['close'], color='blue', label='Close Price')
    #plt.plot(data.index, data['open'], color='red', label='Open Price')

    # Plot support and resistance levels
    for levels_list in levels:
        if levels_list is not None:
            plt.plot(data.index[-len(levels_list):], levels_list, marker='.', linestyle='', color='red')

    # Plot buy signals | sr_signal indicates buy or sell
    plt.plot(data.index[data['sr_signal'] == 1], data['close'][data['sr_signal'] == 1],
            '.', markersize=3, color='green', lw=0, label='Buy Signal')

    # Plot sell signals
    plt.plot(data.index[data['sr_signal'] == -1], data['close'][data['sr_signal'] == -1],
            '.', markersize=3, color='red', lw=0, label='Sell Signal')

    # Customize the plot
    plt.title('Support and Resistance Levels')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    plt.show()

# Runs testing program
def initiate_testing():
    print("Action list:")
    print("1. Test trades")
    print("2. Plot graph")
    selection = int(input("Select a number: "))
    if selection == 1: 
        test_trades()
    elif selection == 2:
        plotting()
    else:
        print("Error")

# initiate_testing() runs program for testing datasets
# test_trades() displays trade info for each currency data file
# plotting() plots price against time along with point of execution of trade including its type
