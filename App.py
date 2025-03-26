import requests
from flask import Flask, request, jsonify, session
from flask_pymongo import PyMongo
from flask_cors import CORS
from datetime import datetime, timedelta, timezone
from get_livePrices import get_price
from get_RSI import fetch_RSI
import time
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model

MODEL_FOLDER = "model"


app = Flask(__name__)
app.config["MONGO_URI"] = "mongodb://localhost:27017/mydatabase"
app.config['SECRET_KEY'] = 'e3c5f6a68e1c4529e929ee8b98207ac9'
mongo = PyMongo(app)
CORS(app)  # Enable CORS for all routes

users_collection = mongo.db.users
prices_collection = mongo.db.prices
candle_chart_prices_collection = mongo.db.candle_chart_prices
bbands_collection = mongo.db.bbands
rsi_collection = mongo.db.rsi
selectedtime_collection = mongo.db.timeFrame
alerts_collection = mongo.db.alerts

Z



# Mapping CoinGecko identifiers correctly
coingecko_coin_ids = {
    "bitcoin": "bitcoin",
    "ethereum": "ethereum",
    "solana": "solana",
    "binancecoin": "binancecoin",
    "ripple": "ripple",
    "dogecoin": "dogecoin",
    "cardano": "cardano",
    "matic-network": "matic",  # CoinGecko uses "matic" for Polygon
    "polkadot": "polkadot"
}


# Function to fetch latest data from CoinGecko
def fetch_coin_data(coin_symbol, days=120):
    # Ensure correct CoinGecko identifier
    if coin_symbol not in coingecko_coin_ids:
        print(f"Invalid coin symbol: {coin_symbol}")
        return None
    
    coingecko_id = coingecko_coin_ids[coin_symbol]
    url = f'https://api.coingecko.com/api/v3/coins/{coingecko_id}/market_chart?vs_currency=usd&days={days}'
    
    response = requests.get(url)
    if response.status_code == 200:
        return response.json().get('prices', [])
    else:
        print(f"Error fetching data for {coin_symbol}: {response.status_code}")
        return None

@app.route('/')
def home():
    return jsonify({"message": "Crypto Price Prediction API is running!"})

@app.route('/predict', methods=['GET'])
def predict_price():
    coin = request.args.get("coin", "").lower()
    
    print(f"Received coin: '{coin}'")  # Debugging output

    if coin not in coingecko_coin_ids:
        return jsonify({"error": "Invalid coin. Available options: " + ", ".join(coingecko_coin_ids.keys())}), 400

    model_path = os.path.join(MODEL_FOLDER, f"{coin}_crypto_price_model.h5")

    if not os.path.exists(model_path):
        return jsonify({"error": f"No trained model found for {coin}"}), 400

    # Load the trained LSTM model
    model = load_model(model_path)

    # Fetch latest 120 days of coin data
    data = fetch_coin_data(coin, days=120)
    if not data:
        return jsonify({"error": f"Failed to fetch data for {coin}"}), 500

    # Convert to DataFrame
    df = pd.DataFrame(data, columns=['timestamp', 'Close'])
    df['Date'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('Date', inplace=True)

    # Preprocess the data
    dataset = df[['Close']].values
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(dataset)

    # Prepare input sequence
    lookback = 120  # Use 4 months of historical data
    x_input = scaled_data[-lookback:].reshape(1, lookback, 1)

    # Predict next 40 days
    future_predictions = []
    for _ in range(40):
        pred = model.predict(x_input)
        future_price = scaler.inverse_transform(pred)[0][0]
        future_predictions.append(round(future_price, 2))

        # Update input sequence with new predicted value
        pred = pred.reshape(1, 1, 1)  # Ensure pred matches LSTM input shape
        x_input = np.append(x_input[:, 1:, :], pred, axis=1)


    return jsonify({
        "coin": coingecko_coin_ids[coin],
        "predicted_prices": [float(pred) for pred in future_predictions]  # Convert float32 to Python float
    })



@app.route('/signup', methods=['POST'])
def signup():
    data = request.get_json()

    username = data.get('username')
    email = data.get('email')
    password = data.get('password')

    if not username or not email or not password:
        return jsonify({'error': 'Provide all credentials'}), 400

    if users_collection.find_one({'email': email}):
        return jsonify({'error': 'Email is already registered'}), 400

    if users_collection.find_one({'username': username}):
        return jsonify({'error': 'Username is already taken'}), 400

    if len(password) < 8:
        return jsonify({'error': 'Password should be at least 8 characters long'}), 400

    users_collection.insert_one({
        'username': username,
        'email': email,
        'password': password
    })

    return jsonify({'message': 'User signed up successfully'}), 201

@app.route('/login', methods=['POST'])
def login():
    data = request.get_json()

    email_or_username = data.get('email_or_username')
    password = data.get('password')

    if not email_or_username or not password:
        return jsonify({'error': 'Provide email/username and password'}), 400

    user = users_collection.find_one({
        '$or': [{'email': email_or_username}, {'username': email_or_username}]
    })

    if not user or user['password'] != password:
        return jsonify({'error': 'Invalid email/username or password'}), 400
    
     # Set session data
    session['username'] = user['username']
    session['email'] = user['email']

    return jsonify({'message': 'User logged in successfully'}), 200

@app.route('/get_user', methods=['GET'])
def get_user():
    # Assuming you have session management
    if 'username' in session:
        user = users_collection.find_one({'username': session['username']})
        if user:
            return jsonify({
                'username': user['username'],
                'email': user['email']
            }), 200
        else:
            return jsonify({'error': 'User not found'}), 404
    else:
        return jsonify({'error': 'Not authenticated'}), 401


@app.route('/logout', methods=['POST'])
def logout():
    session.pop('username', None)
    session.pop('email', None)
    return jsonify({'message': 'Logged out successfully'}), 200


@app.route('/delete_account', methods=['DELETE'])
def delete_account():
    data = request.get_json()
    password = data.get('password')

    # Ensure the user is authenticated
    username = session.get('username')
    email = session.get('email')

    if not username or not email:
        return jsonify({'error': 'User not authenticated'}), 401

    # Verify password
    user = users_collection.find_one({'username': username, 'email': email})
    if not user or user['password'] != password:
        return jsonify({'error': 'Invalid password'}), 403

    # Delete user from the database
    result = users_collection.delete_one({'username': username, 'email': email})
    
    if result.deleted_count == 0:
        return jsonify({'error': 'User not found'}), 404

    # Clear the session
    session.pop('username', None)
    session.pop('email', None)

    return jsonify({'message': 'Account deleted successfully'}), 200



@app.route('/get_live_price', methods=['GET'])
def get_live_price():
    coin = request.args.get('coin')
    if not coin:
        return jsonify({'error': 'No coin specified'}), 400

    price = get_price(coin.upper())
    if price is not None:
        current_time = datetime.now()
        prices_collection.insert_one({
            'coin': coin,
            'price': price,
            'timestamp': current_time
        })

        # Aggregate prices for the candlestick chart
        minute_start = current_time.replace(second=0, microsecond=0)
        candle = candle_chart_prices_collection.find_one({
            'coin': coin,
            'timestamp': minute_start
        })

        if candle:
            # Update existing candle
            candle_chart_prices_collection.update_one(
                {'_id': candle['_id']},
                {'$set': {
                    'close': price,
                    'high': max(candle['high'], price),
                    'low': min(candle['low'], price)
                }}
            )
        else:
            # Create new candle
            candle_chart_prices_collection.insert_one({
                'coin': coin,
                'timestamp': minute_start,
                'open': price,
                'close': price,
                'high': price,
                'low': price
            })

        return jsonify({'coin': coin, 'price': price}), 200
    else:
        return jsonify({'error': 'Failed to retrieve the price'}), 500

@app.route('/api/candlestick', methods=['GET'])
def get_candlestick_data():
    symbol = request.args.get('symbol', default='BTC', type=str)
    limit = request.args.get('limit', default=100, type=int)
    
    # CryptoCompare API endpoint for historical data
    url = 'https://min-api.cryptocompare.com/data/v2/histoday'
    
    # Parameters for CryptoCompare API
    params = {
        'fsym': symbol,  # The cryptocurrency symbol (e.g., BTC)
        'tsym': 'USD',    # The currency to convert to (e.g., USD)
        'limit': limit,   # Number of data points to return
        'api_key': '04d842bc95505e37fc0bd2d279ab5ef5d693103c645fc05bc78c50b96a29177e'
    }
    
    try:
        # Make a request to the CryptoCompare API
        response = requests.get(url, params=params)
        data = response.json()
        
        # Check if the API returned data successfully
        if data.get('Response') == 'Success':
            # Return the candlestick data in the desired format
            candle_data = [
                {
                    'timestamp': item['time'],  # Unix timestamp in seconds
                    'open': item['open'],
                    'high': item['high'],
                    'low': item['low'],
                    'close': item['close']
                }
                for item in data['Data']['Data']
            ]
            return jsonify({'candle_data': candle_data}), 200
        else:
            return jsonify({'error': 'Error fetching data'}), 400
    
    except Exception as e:
        # Handle any errors that occur during the API request
        return jsonify({'error': str(e)}), 500

@app.route('/get_bbands', methods=['GET'])
def get_bbands():
    api_key = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJjbHVlIjoiNjZiNjEwMTExNjY1YTQ1MTMyZjVhOTgxIiwiaWF0IjoxNzIzMjA3Njk3LCJleHAiOjMzMjI3NjcxNjk3fQ.m5pkS72gAH2LCaNYU01JRsS6yNNS0V-EHv3ZJzbH8QY'
    exchange = 'binance'
    symbol = request.args.get('symbol')
    interval = request.args.get('interval')
    
    api_url = f"https://api.taapi.io/bbands?secret={api_key}&exchange={exchange}&symbol={symbol}&interval={interval}"
    
    try:
        response = requests.get(api_url)
        
        # Check if the request was successful
        if response.status_code == 200:
            data = response.json()
            print("Response data:", data)  
            upper_band = data.get('valueUpperBand')
            middle_band = data.get('valueMiddleBand')
            lower_band = data.get('valueLowerBand')

            bbands_collection.insert_one({
                'symbol': symbol,
                'interval': interval,
                'upper_band': upper_band,
                'middle_band': middle_band,
                'lower_band': lower_band,
                'timestamp': datetime.utcnow()
            })

            return jsonify({
                "upper_band": upper_band,
                "middle_band": middle_band,
                "lower_band": lower_band
            }), 200
        else:
            print(f"Error: Unable to fetch data. Status code: {response.status_code}")
            return jsonify({"error": "Failed to fetch Bollinger Bands data"}), response.status_code
    except Exception as e:
        print(f"An error occurred: {e}")
        return jsonify({"error": "An internal error occurred"}), 500

@app.route('/get_rsi', methods=['GET'])
def get_rsi():
    coin = request.args.get('coin')
    interval = request.args.get('interval')
    
    if not coin or not interval:
        return jsonify({'error': 'Coin or interval not specified'}), 400
    
    rsi_value = fetch_RSI(coin, interval)
    
    if rsi_value is not None:
        rsi_data = {
            "coin": coin,
            "value": rsi_value,
            "timestamp": datetime.utcnow()
        }
        rsi_collection.insert_one(rsi_data)
        return jsonify({'message': 'RSI data fetched and stored successfully', 'rsi_value': rsi_value}), 200
    else:
        return jsonify({'error': 'RSI value not found or error fetching data'}), 500

@app.route('/save_time', methods=['POST'])
def save_time():
     # Retrieve user information from session
    username = session.get('username')
    email = session.get('email')
    
    if not username or not email:
        return jsonify({'error': 'User not authenticated'}), 401
    
    data = request.get_json()
    timeFrame = data.get('selectedTime')
    symbol = data.get('selectedValue')

    if not timeFrame or not symbol:
        return jsonify({'error': 'Time interval and symbol are required'}), 400

    try:
        time_interval_minutes = int(timeFrame.rstrip('m'))
        start_time = datetime.utcnow()
        end_time = start_time + timedelta(minutes=time_interval_minutes)
    except ValueError:
        return jsonify({'error': 'Invalid time interval provided'}), 400
    
    new_time = {
        "timeFrame": timeFrame,
        "symbol": symbol,
        "start_time": start_time,
        "end_time": end_time,
        "created_at": start_time,
        "username": username, 
        "email": email      
    }
    selectedtime_collection.insert_one(new_time)

    return jsonify({'message': 'Time frame saved successfully for the selected coin'}), 201


@app.route('/generate_alerts', methods=['POST'])
def generate_alerts():
    # Retrieve user information from session
    username = session.get('username')
    email = session.get('email')
    
    if not username or not email:
        return jsonify({'error': 'User not authenticated'}), 401
    
    data = request.get_json()
    timeFrame = data.get('selectedTime')
    symbol = data.get('selectedValue')

    if not timeFrame or not symbol:
        return jsonify({'error': 'Time interval and symbol are required'}), 400

    try:
        time_interval_minutes = int(timeFrame.rstrip('m'))
        start_time = datetime.utcnow()
        end_time = start_time + timedelta(minutes=time_interval_minutes)
    except ValueError:
        return jsonify({'error': 'Invalid time interval provided'}), 400

    while datetime.utcnow() < end_time:
        rsi_value = fetch_RSI(symbol, timeFrame)
        
        if rsi_value is None:
            return jsonify({'error': 'RSI value not found or error fetching data'}), 500

        alert_message = None
        if 25 < rsi_value < 35:
            alert_message = 'Buy the coin'
        elif 40 < rsi_value < 75:
            alert_message = 'Sell the coin'

        if alert_message:
            alert_data = {
                "timeFrame": timeFrame,
                "start_time": start_time,
                "end_time": end_time,
                "symbol": symbol,
                "created_at": datetime.utcnow(),
                "rsi_value": rsi_value,
                "message": alert_message,
                "username": username, 
                "email": email   
            }
            alerts_collection.insert_one(alert_data)
        
        time.sleep(15)

    return jsonify({'message': 'Alert generation process completed.'}), 201

@app.route('/get_alerts', methods=['GET'])
def get_alerts():
    try:
        timeFrame = request.args.get('selectedTime')
        symbol = request.args.get('selectedValue')

        if not timeFrame or not symbol:
            return jsonify({'error': 'Time interval and symbol are required'}), 400

        try:
            time_interval_minutes = int(timeFrame.rstrip('m')) # Extract the number of minutes from the timeFrame string
            current_time = datetime.utcnow()
            start_time = current_time - timedelta(minutes=time_interval_minutes) # Calculate the start time by subtracting the timeFrame (interval) from the current time
        except ValueError:
            return jsonify({'error': 'Invalid time interval provided'}), 400

        alerts = alerts_collection.find(
            {
                'symbol': symbol,         # Match the symbol
                'created_at': {           # Find alerts within the specified time range
                    '$gte': start_time,   # Greater than or equal to start_time
                    '$lte': current_time  # Less than or equal to end_time
                }
            },
            {
                'symbol': 1,       
                'timeFrame': 1,
                'message': 1,
                'rsi_value': 1,
                'username': 1,
                'email': 1,
                'created_at' : 1,
                '_id': 0          
            }
        )

        alerts_list = list(alerts)   # Convert the result to a list
        
        return jsonify({'alerts': alerts_list}), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500
   
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
