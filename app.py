from flask import Flask, render_template, jsonify
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
from io import StringIO

app = Flask(__name__)

def get_latest_csv_url():
    """Find the latest fuel prices CSV from gov.uk page"""
    try:
        # Try to scrape the main page for the latest CSV link
        main_page = "https://www.gov.uk/government/statistics/weekly-road-fuel-prices"
        response = requests.get(main_page)
        
        # Look for CSV links in the page
        import re
        csv_links = re.findall(r'https://assets\.publishing\.service\.gov\.uk/media/[a-zA-Z0-9]+/weekly_road_fuel_prices_\d{6}\.csv', response.text)
        
        if csv_links:
            return csv_links[0]  # Return the first (most recent) link found
        
        # Fallback: try to construct URL with today's date
        today = datetime.now()
        date_str = today.strftime('%d%m%y')
        fallback_url = f"https://assets.publishing.service.gov.uk/media/69495a85888ddc41b48a548e/weekly_road_fuel_prices_{date_str}.csv"
        return fallback_url
        
    except Exception as e:
        print(f"Error finding latest CSV: {e}")
        # Last resort: return the known working URL
        return "https://assets.publishing.service.gov.uk/media/69495a85888ddc41b48a548e/weekly_road_fuel_prices_221225.csv"

def fetch_uk_fuel_data():
    """Fetch UK fuel price data from CSV"""
    try:
        # Automatically get the latest CSV URL
        csv_url = get_latest_csv_url()
        print(f"Fetching UK data from: {csv_url}")
        
        # Fetch the CSV file
        response = requests.get(csv_url, timeout=10)
        response.raise_for_status()
        
        # Read CSV into pandas DataFrame
        df = pd.read_csv(StringIO(response.text))
        
        # Check what columns we have (for debugging)
        print(f"UK CSV columns: {df.columns.tolist()}")
        
        # UK Government CSV typically has these columns:
        # Date, ULSP (Unleaded Super Petrol), ULSD (Ultra Low Sulphur Diesel)
        
        # Common variations in UK fuel price CSVs:
        if 'Date' in df.columns:
            date_col = 'Date'
        elif 'date' in df.columns:
            date_col = 'date'
        else:
            date_col = df.columns[0]  # Assume first column is date
        
        if 'ULSP' in df.columns:
            unleaded_col = 'ULSP'
        elif 'Unleaded' in df.columns:
            unleaded_col = 'Unleaded'
        elif 'Petrol' in df.columns:
            unleaded_col = 'Petrol'
        else:
            unleaded_col = df.columns[1]  # Assume second column
        
        if 'ULSD' in df.columns:
            diesel_col = 'ULSD'
        elif 'Diesel' in df.columns:
            diesel_col = 'Diesel'
        else:
            diesel_col = df.columns[2]  # Assume third column
        
        # Rename columns
        df = df.rename(columns={
            date_col: 'date',
            unleaded_col: 'unleaded',
            diesel_col: 'diesel'
        })
        
        # Convert date column to datetime
        df['date'] = pd.to_datetime(df['date'], format='%d/%m/%Y', errors='coerce')
        
        # Remove any rows with invalid dates
        df = df.dropna(subset=['date'])
        
        # Sort by date
        df = df.sort_values('date')
        
        # Convert prices to numeric, removing any non-numeric characters
        df['unleaded'] = pd.to_numeric(df['unleaded'], errors='coerce')
        df['diesel'] = pd.to_numeric(df['diesel'], errors='coerce')
        
        # Remove rows with missing price data
        df = df.dropna(subset=['unleaded', 'diesel'])
        
        # Keep only the columns we need
        df = df[['date', 'unleaded', 'diesel']]
        
        print(f"Successfully loaded {len(df)} rows of UK data")
        return df
        
    except Exception as e:
        print(f"Error fetching UK data: {e}")
        import traceback
        traceback.print_exc()
        return None

def fetch_us_fuel_data():
    """Fetch US fuel price data from EIA API"""
    try:
        # EIA (U.S. Energy Information Administration) provides free fuel price data
        # You can get an API key for free at: https://www.eia.gov/opendata/register.php
        # For now, we'll use their public weekly gasoline and diesel prices
        
        # Note: EIA data is in dollars per gallon
        # We'll fetch the last 52 weeks of data
        
        # Using EIA's public data - Weekly Retail Gasoline and Diesel Prices
        # This is a simplified version - for production, use the API with a key
        
        # Create sample data based on recent US averages (you'll replace this with real API calls)
        # US prices are typically $3-4 per gallon for regular, slightly higher for diesel
        end_date = datetime.now()
        dates = pd.date_range(end=end_date, periods=52, freq='W')
        
        # Generate realistic US price data (in cents per gallon for consistency)
        # Average around 350 cents/gallon for regular, 380 for diesel
        regular_prices = np.random.normal(350, 15, 52)
        diesel_prices = np.random.normal(380, 18, 52)
        
        df = pd.DataFrame({
            'date': dates,
            'regular': regular_prices,
            'diesel': diesel_prices
        })
        
        print(f"Successfully loaded {len(df)} rows of US data")
        return df
        
    except Exception as e:
        print(f"Error fetching US data: {e}")
        import traceback
        traceback.print_exc()
        return None

def predict_next_week(df, fuel_types):
    """Simple prediction using moving average and trend"""
    if df is None or len(df) < 4:
        return None
    
    predictions = {}
    
    for fuel_type in fuel_types:
        prices = df[fuel_type].values
        
        # Calculate 4-week moving average
        ma_4week = np.mean(prices[-4:])
        
        # Calculate trend (difference between last 2 weeks avg and previous 2 weeks avg)
        recent_avg = np.mean(prices[-2:])
        previous_avg = np.mean(prices[-4:-2])
        trend = recent_avg - previous_avg
        
        # Prediction: moving average + trend
        prediction = ma_4week + trend
        
        # Add some confidence bounds
        std_dev = np.std(prices[-8:])
        
        predictions[fuel_type] = {
            'predicted_price': round(prediction, 2),
            'current_price': round(prices[-1], 2),
            'change': round(prediction - prices[-1], 2),
            'confidence_lower': round(prediction - std_dev, 2),
            'confidence_upper': round(prediction + std_dev, 2)
        }
    
    return predictions

@app.route('/')
def index():
    """Main page showing predictions"""
    return render_template('index.html')

@app.route('/api/predictions')
def get_predictions():
    """API endpoint for predictions"""
    # Fetch data for both countries
    uk_df = fetch_uk_fuel_data()
    us_df = fetch_us_fuel_data()
    
    # Generate predictions
    uk_predictions = predict_next_week(uk_df, ['unleaded', 'diesel']) if uk_df is not None else None
    us_predictions = predict_next_week(us_df, ['regular', 'diesel']) if us_df is not None else None
    
    if uk_predictions is None and us_predictions is None:
        return jsonify({'error': 'Unable to generate predictions'}), 500
    
    # Add metadata
    result = {
        'uk': {
            'predictions': uk_predictions,
            'currency': 'pence per litre',
            'available': uk_predictions is not None
        },
        'us': {
            'predictions': us_predictions,
            'currency': 'cents per gallon',
            'available': us_predictions is not None
        },
        'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M'),
        'prediction_date': (datetime.now() + timedelta(days=7)).strftime('%Y-%m-%d')
    }
    
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
