import oandapyV20
import oandapyV20.endpoints.pricing as pricing

api_token = "your_api_token_here"
client = oandapyV20.API(access_token=api_token, environment="practice")

params = {"instruments": "EUR_USD"}
r = pricing.PricingInfo(accountID="your_account_id_here", params=params)

response = client.request(r)
print(response["prices"][0]["bids"][0]["price"])