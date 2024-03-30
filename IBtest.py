#imports
import ibapi
from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.contract import Contract
from ibapi.order import *
import threading
import time

#classes

#Class for IB Connection
class IBApi(EWrapper, EClient):
    def _init_(self):
        EClient._init_(self, self)

    #Listen for real time bars
    def realtimeBar(self, reqId, time, open_, high, low, close, volume, wap, count):
        bot.on_bar_update(reqId, time, open_, high, low, close, volume, wap, count)

#Class for Bot
class Bot():
    ib = None
    #Connects to IB on init
    def _init_(self):
        ib = IBApi()
        ib.connect("127.0.0.1", 7497, clientId=1)
        ib_thread = threading.Thread(target = self.run_loop, deamon = True)
        ib_thread.start()
        time.sleep(1)
        #Get Symbol Info
        symbol = input("Enter symbol you want to trade: ")

        #Create IB Contract Object
        contract = Contract()
        contract.symbol = symbol.upper()
        contract.secType = "CASH"
        contract.exchange = "SMART"
        contract.currency = "USD"

        #Request Market Data
        self.ib.reqRealTimeBars(0, contract, 5, "TRADES", 1, [])

        #Create Order Object
        order = Order()
        order.orderType = "MKT"
        order.action = "BUY"
        quantity = 1
        order.totalQuantity = quantity

        #Create Contract Object
        contract = Contract()
        contract.symbol = symbol
        contract.secType = "CASH"
        contract.exchange = "SMART"
        contract.primaryExchange = "ISLAND"
        contract.currency = "USD"

        #Place the order
        self.ib.placeOrder(1, contract, order)
        
    #listen to socket in seperate thread
    def run_loop(self):
        self.ib.run()

    #Pass real time bar data back to bot object
    def on_bar_update(self, reqId, time, open_, high, low, close, volume, wap, count):
        print(close)

#Start Bot
bot = Bot()

