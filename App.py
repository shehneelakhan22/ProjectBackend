import requests
from flask import Flask, request, jsonify, session
from flask_pymongo import PyMongo
from flask_cors import CORS
from datetime import datetime, timedelta, timezone
from get_livePrices import get_price
from get_RSI import fetch_RSI
import time
import yfinance as yf
import os
from flask_socketio import SocketIO, emit # type: ignore
from threading import Thread
import numpy as np  
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model # type: ignore
from datetime import datetime, timedelta
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer # type: ignore
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from datetime import datetime


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
wallets_collection = mongo.db.wallets




############################################################################################

@app.route('/update_wallet', methods=['POST'])
def update_wallet():
    if 'username' not in session:
        return jsonify({'error': 'User not authenticated'}), 401

    data = request.get_json()
    username = session['username']

    investmentAmount = data.get('amount')                # Investment when user starts the bot
    print(f"Investment Amount: {investmentAmount}")

    currentUpdatedBalance = data.get('currentBalance')   # Total amount when user stops the bot

    # Validate inputs
    if investmentAmount is None and currentUpdatedBalance is None:
        return jsonify({'error': 'Provide either "amount" or "currentBalance"'}), 400

    # Fetch current wallet
    wallet = wallets_collection.find_one({'username': username})
    current_amount = float(wallet.get('balance', 0)) if wallet else 0

    if currentUpdatedBalance is not None:
        if not isinstance(currentUpdatedBalance, (int, float)):
            return jsonify({'error': '"currentBalance" must be a number'}), 400
        updated_balance = currentUpdatedBalance

    # If 'amount' is provided, add to the current balance
    elif investmentAmount is not None:
        if not isinstance(investmentAmount, (int, float)):
            return jsonify({'error': '"amount" must be a number'}), 400
        updated_balance = current_amount + investmentAmount

    # Format balance with thousand separator
    formatted_balance = f"{updated_balance:,.2f}"  # e.g., 1000 -> '1,000.00'

    wallets_collection.update_one(
        {'username': username},
        {
            '$set': {
                'balance': formatted_balance,
                'currency': 'USD',
                'last_updated': datetime.utcnow()
            }
        },
        upsert=True
    )

    return jsonify({'message': f'Wallet updated successfully. New balance: ${updated_balance:.2f}'}), 200


# Get wallet amount
@app.route('/get_balance', methods=['GET'])
def get_balance():
    if 'username' not in session:
        return jsonify({'error': 'User not authenticated'}), 401

    username = session['username']
    wallet = wallets_collection.find_one({'username': username})

    if not wallet:
        return jsonify({'error': 'Wallet not found'}), 404

    # Get the balance from the wallet and ensure it's a string with a thousand separator
    balance = wallet.get('balance', 0)
    formatted_balance = "{:,.2f}".format(balance)  # This will add the thousand separator

    return jsonify({'balance': formatted_balance}), 200


############################################################################################



# Load pre-trained models
model_directory = "trainedmodels"
models = {
    'bitcoin': load_model(os.path.join(model_directory, 'bitcoin_model.h5')),
    'binancecoin': load_model(os.path.join(model_directory, 'binancecoin_model.h5')),
    'ethereum': load_model(os.path.join(model_directory, 'ethereum_model.h5')),
    'solana': load_model(os.path.join(model_directory, 'solana_model.h5'))
}

# Constants for News API
NEWS_API_KEY = 'c01410fe218b4e299d30e04d2a6ff53c'

# Fetch cryptocurrency data from CoinGecko API
def fetch_coin_data(coin_symbol, days=365):
    url = f'https://api.coingecko.com/api/v3/coins/{coin_symbol}/market_chart?vs_currency=usd&days={days}'
    response = requests.get(url)
    if response.status_code == 200:
        return response.json().get('prices', [])
    return []

# Fetch news articles relevant to the entire crypto market (macro-level)
def fetch_crypto_news():
    query = "cryptocurrency OR bitcoin OR ethereum OR crypto regulation OR adoption OR crash OR ETF OR market"
    url = f"https://newsapi.org/v2/everything?q={query}&language=en&sortBy=publishedAt&apiKey={NEWS_API_KEY}"
    response = requests.get(url)
    if response.status_code == 200:
        return [article['title'] for article in response.json().get('articles', [])]
    return []

# Analyze sentiment of news headlines using VADER
def analyze_sentiment(texts):
    analyzer = SentimentIntensityAnalyzer()
    scores = [analyzer.polarity_scores(t)['compound'] for t in texts]
    return np.mean(scores) if scores else 0

# Function to predict the next 40 days of cryptocurrency price
def make_40_days_prediction(coin_symbol, sentiment_score, lookback=60):
    # Fetch cryptocurrency data (last 365 days)
    data = fetch_coin_data(coin_symbol, days=365)
    if not data:
        return {"error": "Failed to fetch data for the coin."}
    
    df = pd.DataFrame(data, columns=['timestamp', 'Close'])
    df['Date'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('Date', inplace=True)
    
    # Add sentiment score to the dataset
    df['Sentiment'] = sentiment_score

    # Prepare data for prediction
    dataset = df[['Close', 'Sentiment']].values
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(dataset)

    # Create sequences for training (ensure 3D shape)
    x_input = scaled_data[-lookback:]
    x_input = np.reshape(x_input, (1, lookback, 2))  # Shape: (1, lookback, 2)
    
    # Make predictions for the next 40 days
    future_predictions = []
    for _ in range(40):
        pred = models[coin_symbol].predict(x_input)[0][0]
        future_predictions.append(float(pred))  # Convert to float
        new_entry = np.array([[pred, sentiment_score]])
        x_input = np.vstack((x_input[0][1:], new_entry))
        x_input = np.reshape(x_input, (1, lookback, 2))  # Keep the shape consistent

    # Rescale the predictions to original price scale
    future_scaled = np.hstack((np.array(future_predictions).reshape(-1, 1), np.zeros_like(np.array(future_predictions).reshape(-1, 1))))
    future_prices = scaler.inverse_transform(future_scaled)[:, 0]

    # Convert future prices to float and round
    future_prices = [round(float(price), 2) for price in future_prices]

    # Generate future dates
    future_dates = pd.date_range(start=df.index[-1] + timedelta(days=1), periods=40)

    # Return the results
    return {
        "future_dates": [date.strftime('%Y-%m-%d') for date in future_dates],
        "predictions": future_prices  # Return the predictions as a list of floats
    }

# Route to fetch latest market-wide cryptocurrency news
@app.route('/news', methods=['GET'])
def get_crypto_news():
    news_headlines = fetch_crypto_news()
    if news_headlines:
        return jsonify({"headlines": news_headlines})
    else:
        return jsonify({"error": "Failed to fetch news."}), 500

# Route to predict cryptocurrency prices for the next 40 days
@app.route('/predict', methods=['GET'])
def predict():
    coin_symbol = request.args.get('coin', type=str).lower()
    if coin_symbol not in models:
        return jsonify({"error": "Invalid coin symbol. Available coins: bitcoin, binancecoin, ethereum, solana."})

    # Fetch market-wide news and analyze sentiment
    news_headlines = fetch_crypto_news()
    sentiment_score = analyze_sentiment(news_headlines)

    # Predict the next 40 days
    prediction_result = make_40_days_prediction(coin_symbol, sentiment_score)
    
    return jsonify(prediction_result)


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
    
    user_result = users_collection.insert_one({
    'username': username,
    'email': email,
    'password': password
       })

    user_id = str(user_result.inserted_id)

    wallets_collection.insert_one({
    'user_id': user_id,
    'username': username,
    'balance': 0.0,  # USD
    'created_at': datetime.utcnow()
    })

    return jsonify({'message': 'User signed up successfully and wallet created'}), 201


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
   
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
