import requests

def fetch_RSI(coin, interval):
    api_key = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJjbHVlIjoiNjY4M2U1NzBmNWFmOTRlZWNlNmJiNjU3IiwiaWF0IjoxNzIzMTE0MTc1LCJleHAiOjMzMjI3NTc4MTc1fQ.5Hh4JoxFUpOWVVY6eUQ30WJd7io5tp88Jj3bAHTixXQ'  
    exchange = 'binance'
    symbol = f"{coin}/USDT"
    api_url = f"https://api.taapi.io/rsi?secret={api_key}&exchange={exchange}&symbol={symbol}&interval={interval}"
    
    try:
        response = requests.get(api_url)
        response.raise_for_status()
        data = response.json()
        rsi_value = data.get('value')

        if rsi_value is not None:
            return rsi_value
        else:
            print("RSI value is None.")
            return None
    except requests.RequestException as e:
        print(f"Error fetching RSI from API: {e}")
        return None
