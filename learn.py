import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy
import pandas_ta as ta
from matplotlib.ticker import MaxNLocator

# first_w: float = 0.1, atr_mult:float=1.0, prom_thresh:float =0.05
# first_w varies weightage of prices depending on length of time from present
# atr_mult affects smoothing of KDE
# As prom_thresh >, only resistance levels with greater significance taken 

# Array creation to store iteration data
profit_map = []
first_w_map = []
atr_mult_map = []
prom_thresh_map = []

# Variable declaration
first_w = 0.1
atr_mult = 1.0
prom_thresh = 0.05
profit = 0
count = 0

csv_storage = {
    "EURUSD": "D:\Downloads\EURUSD_Candlestick_1_D_BID_05.05.2003-28.10.2023.csv",
    "BTCUSDT": "D:\Downloads\BTCUSDT86400.csv",
    "TESLA": "D:\Downloads\esla-stock-price.csv",
}

# Prints detailed output for single iteration
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


# Plots data
def single_plot():
    # Plot closing prices
    plt.figure(figsize=(20, 6))
    plt.plot(data.index, data['close'], color='blue', label='Close Price')
    plt.plot(data.index, data['open'], color='red', label='Open Price')

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


# Plots total profit against iteration count
def sum_plot():
    global profit_map
    plt.figure(figsize=(15, 20))
    plt.plot(range(1, len(profit_map) + 1), profit_map, marker='.', linestyle='', color='red', label='Profit')
    
    plt.title('Profit vs Iteration')
    plt.xlabel('Iteration')
    plt.ylabel('Profit')
    plt.legend()
    plt.grid(True)
    # Set the y-axis to use whole numbers
    plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))
    plt.ylim(min(profit_map), max(profit_map))
    plt.show()


# Logs iteration data into an array
def data_log (profit, first_w, atr_mult, prom_thresh):
    profit_map.append(profit)
    first_w_map.append(first_w)
    atr_mult_map.append(atr_mult)
    prom_thresh_map.append(prom_thresh)
    return profit_map, first_w_map, atr_mult_map, prom_thresh_map

# Displays total profit
def test_trades_adjust():
    global profit
    profit = long_profit + short_profit
    print("Profit: $"+str(profit))


# Prints parameters used to obtain max_profit
def max_array ():
    global profit_map, first_w_map, atr_mult_map, prom_thresh_map
    max_item = profit_map.index(max(profit_map)) + 1
    profit_map = list(map(str, profit_map))
    first_w_map = list(map(str, first_w_map))
    atr_mult_map = list(map(str, atr_mult_map))
    prom_thresh_map = list(map(str, prom_thresh_map))
    print("\nIteration number: " + str(max_item + 1))
    print("Maximum profit: $" + str(profit_map[max_item - 1]))
    print("first_w: " + str(first_w_map[max_item]))
    print("atr_mult_map: " + str(atr_mult_map[max_item]))
    print("prom_thresh: " + str(prom_thresh_map[max_item]))


# iteration loop
while first_w < 0.5:
    first_w += 0.1
    atr_mult = 1.0    
    while atr_mult < 2:
        atr_mult += 0.1
        prom_thresh = 0.05
        while prom_thresh < 0.25:
            prom_thresh += 0.05
            count += 1
            print("Iteration number: " + str(count))
            print("first_w: " + str(first_w))
            print("atr_mult: " + str(atr_mult))
            print("prom_thresh: " + str(prom_thresh) + "\n")
            
            # Log closing price, and log atr
            def find_levels( 
                price: np.array, atr: float,  
                first_w,
                atr_mult ,
                prom_thresh
                ):

            # Setup weights
                last_w = 1.0
                w_step = (last_w - first_w) / len(price)
                weights = first_w + np.arange(len(price)) * w_step
                weights[weights < 0] = 0.0

                # Get kernel of price. 
                # bw_method = bandwidth of KDE
                kernal = scipy.stats.gaussian_kde(price, bw_method = atr * atr_mult, weights = weights)

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

                peaks, props = scipy.signal.find_peaks(pdf, prominence = prom_min)
                levels = [] 
                for peak in peaks:
                    levels.append(np.exp(price_range[peak]))

                return levels, peaks, props, price_range, pdf, weights
            

            def support_resistance_levels(
                data: pd.DataFrame, lookback: int, 
                first_w, atr_mult, prom_thresh
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

                    signal[i] = curr_sig
                    # If signal == 1, buy / If signal == -1, then sell
                return signal


            def get_trades_from_signal(data: pd.DataFrame, signal: np.array):
                long_trades = []
                short_trades = []

                close_arr = data['close'].to_numpy()
                last_sig = 0.0
                open_trade = None
                idx = data.index
                for i in range(len(data)):
                    if signal[i] == 1.0 and last_sig != 1.0: # Long entry
                        if open_trade is not None:
                            open_trade[2] = idx[i] 
                            open_trade[3] = close_arr[i]
                            short_trades.append(open_trade)

                        open_trade = [idx[i], close_arr[i], -1, np.nan]
                    if signal[i] == -1.0  and last_sig != -1.0: # Short entry
                        if open_trade is not None:
                            open_trade[2] = idx[i] 
                            open_trade[3] = close_arr[i]
                            long_trades.append(open_trade)

                        open_trade = [idx[i], close_arr[i], -1, np.nan]

                    last_sig = signal[i]

                long_trades = pd.DataFrame(long_trades, columns=['entry_time', 'entry_price', 'exit_time', 'exit_price'])
                short_trades = pd.DataFrame(short_trades, columns=['entry_time', 'entry_price', 'exit_time', 'exit_price'])

                # lot_size is solely for data logging purposes
                lot_size = float(10000)
                long_trades['percent'] = (long_trades['exit_price'] - long_trades['entry_price']) / long_trades['entry_price'] 
                short_trades['percent'] = -1 * (short_trades['exit_price'] - short_trades['entry_price']) / short_trades['entry_price']
                long_trades['profit'] = long_trades['percent'] * lot_size
                short_trades['profit'] = short_trades['percent'] * lot_size
                long_trades = long_trades.set_index('entry_time')
                short_trades = short_trades.set_index('entry_time')
                return long_trades, short_trades 


            if __name__ == '__main__':

                # Extracts data from .csv file
                data = pd.read_csv("D:\Downloads\EURUSD_Candlestick_1_D_BID_05.05.2003-28.10.2023.csv").tail(400)
                data['date'] = data['date'].astype('datetime64[s]')
                data = data.set_index('date')
                plt.style.use('dark_background') 
                levels = support_resistance_levels(data, 365, first_w = first_w, atr_mult = atr_mult, prom_thresh = prom_thresh)

                data['sr_signal'] = sr_penetration_signal(data, levels)
                data['log_ret'] = np.log(data['close']).diff().shift(-1)
                data['sr_return'] = data['sr_signal'] * data['log_ret']

                long_trades, short_trades = get_trades_from_signal(data, data['sr_signal'].to_numpy())
                # Rounds off calculated profit to 2dp
                long_profit = float(round(sum(long_trades['profit']),2))
                short_profit = float(round(sum(short_trades['profit']),2))
                test_trades_adjust()
                data_log(profit, first_w, atr_mult, prom_thresh)
                
max_array()
sum_plot()


          

      
# Constructs market profile using logarithmic pricing
# first_w varies weightage of prices depending on length of time from present
# atr_mult affects smoothing of KDE
# As prom_thresh >, only resistance levels with greater significance taken 

   #print(data)





