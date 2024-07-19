from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce

# Initialise API info
apikey = 'PKYJ9M5NKMTQSJYSPFR6'
secretkey = '7unbBaA545B7bFPb8Zp3sHPtyci0IgqR9BT2jNyu'
trading_client = TradingClient(apikey, secretkey, paper=True)

signal = 1.0

if signal==1.0:
    # Preparing BUY order data
    market_order_data = MarketOrderRequest(
        symbol="TSLA",
        qty=1,
        side=OrderSide.BUY,
        time_in_force=TimeInForce.DAY
    )

    try:
        market_order = trading_client.submit_order(
        order_data=market_order_data
        )
        print("Order submitted successfully:", market_order)
    except Exception as e:
        print("Error submitting order:", e)
    
elif signal==-1.0:
    # Preparing SELL order data
    market_order_data = MarketOrderRequest(
        symbol="TSLA",
        qty=1,
        side=OrderSide.SELL,
        time_in_force=TimeInForce.DAY
    )

    try:
        market_order = trading_client.submit_order(
        order_data=market_order_data
        )
        print("Order submitted successfully:", market_order)
    except Exception as e:
        print("Error submitting order:", e)

else:
     print("Holding")


