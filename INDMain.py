import pandas as pd 
import pandas_ta as ta
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy import stats

df = pd.read_csv("EURUSD_Candlestick_1_Hour_BID_04.05.2020-15.04.2023.csv")
df = df[df['volume']!=0]
df.reset_index(drop=True, inplace=True)

df['EMA'] = ta.ema(df.close, length=150)
df.tail()

df=df[0:5000]


#Trend detection

EMAsignal = [0]*len(df)
backcandles = 15
for row in range(backcandles, len(df)):
    upt = 1
    dnt = 1
    for i in range(row-backcandles, row+1):
        if max(df.open[i], df.close[i])>=df.EMA[i]:
            dnt = 0
        if min(df.open[i], df.close[i])<=df.EMA[i]:
            upt = 0
    if upt==1 and dnt==1:
        EMAsignal[row]=3
    elif upt==1:
        EMAsignal[row]=2
    elif dnt==1:
        EMAsignal[row]=1

df['EMASignal'] = EMAsignal

def isPivot(candle, window):
    # function that detects if a candle is a pivot/fractal point
    # args: candle index, window before and after candle to test if pivot
    # returns: 1 if pivot high, 2 if pivot low, 3 if both and a default
    if candle-window < 0 or candle+window >= len(df):
        return 0
    
    pivotHigh = 1
    pivotLow = 2
    for i in range(candle-window, candle+window)+1:
        if df.iloc[candle].low > df.iloc[i].low:
            pivotLow = 0
        if df.iloc[candle].high < df.iloc[i].high:
            pivotHigh = 0
    if (pivotHigh and pivotLow):
        return 3
    elif pivotHigh:
        return pivotHigh
    elif pivotLow:
        return pivotLow
    else:
        return 0
    

# window size
window = 10
df['isPivot'] = df.apply(lambda x: isPivot(x.name,window), axis=1)

def pointpos(x):
    if x['isPivot'] == 2:
        return x['low']-1e-3
    elif x['isPivot'] == 1:
        return x['isPivot']+le-3
    else:
        return np.nan
df['pointpos'] = df.apply(lambda row: pointpos(row), axis=1)


# new section
dfpl = df[300:500]
fig = go.figure(data=[go.Candlestick(x=dfpl.index,
                open=dfpl['open'],
                high=dfpl['high'],
                low=dfpl['low'],
                close=dfpl['close'])])

fig.add_scatter(x=dfpl.index, y=dfpl['pointpos'], mode="markers",
                marker=dict(size=5, color="MediumPurple"),
                name="pivot")
fig.update_layout(xaxis_rangeslider_visible=False)
fig.show()

