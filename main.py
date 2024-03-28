import alpaca_trade_api as tradeapi
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce

# Replace these values with your Alpaca API credentials
API_KEY = 'PKP9E58QBX2XLVD0AJT9'
API_SECRET = 'lUyfHlOZ9fbW8oZRDqV2r52ZIyvQyQTOWemvRxYP'
BASE_URL = 'https://paper-api.alpaca.markets/v2'

# Initialize the Alpaca API client
api = tradeapi.REST(API_KEY, API_SECRET, BASE_URL, api_version='v2')

# Define the symbol for GBPUSD
symbol = 'AAPL'

# Get the latest quote data for GBPUSD
latest_quote = api.get_latest_trade(symbol)

# Print the latest quote data
print("Latest AAPL Quote:")
print("Price:", latest_quote.price)
print("Timestamp:", latest_quote.timestamp)

trading_client = TradingClient(API_KEY, API_SECRET, paper=True)
"""
# preparing market order
market_order_data = MarketOrderRequest(
                    symbol="AAPL",
                    qty=0.1,
                    side=OrderSide.BUY,
                    time_in_force=TimeInForce.DAY
                    )

# Market order
market_order = trading_client.submit_order(
                order_data=market_order_data
               )

#sell positions
trading_client.close_position('AAPL')
"""