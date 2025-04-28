import requests

# Replace 'your_api_key' with your actual LiveCoinWatch API key
API_KEY = '41aab9f3-7595-4b10-8110-492d83d03c36'
BASE_URL = 'https://api.livecoinwatch.com/coins/single'

def get_price(coin):
    headers = {
        'content-type': 'application/json',
        'x-api-key': API_KEY,
    }
    payload = {
        'currency': 'USD',
        'code': coin,
        'meta': True
    }
    response = requests.post(BASE_URL, headers=headers, json=payload)
    
    if response.status_code == 200:
        data = response.json()
        return data['rate']
    else:
        return None
