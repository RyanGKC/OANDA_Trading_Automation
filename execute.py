from INDMain import plotting, get_trades_from_signal, sr_penetration_signal, support_resistance_levels, find_levels
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mplfinance as mpf
import scipy
import math
import pandas_ta as ta

# Plug in AV input into data
# long_trades, short_trades = get_trades_from_signal(data, data['sr_signal'].to_numpy())
