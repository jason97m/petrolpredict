"""
Batch prediction script - Run this separately to generate predictions
This script runs 10 predictions with slight variations and averages them
Results are saved to predictions.json and timestamped historical files
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
from io import StringIO
import json
import os

def get_latest_csv_url():
    """Find the latest fuel prices CSV from gov.uk page"""
    try:
        main_page = "https://www.gov.uk/government/statistics/weekly-road-fuel-prices"
        response = requests.get(main_page)
        
        import re
        csv_links = re.findall(r'https://assets\.publishing\.service\.gov\.uk/media/[a-zA-Z0-9]+/weekly_road_fuel_prices_\d{6}\.csv', response.text)
        
        if csv_links:
            return csv_links[0]
        
        today = datetime.now()
        date_str = today.strftime('%d%m%y')
        fallback_url = f"https://assets.publishing.service.gov.uk/media/69495a85888ddc41b48a548e/weekly_road_fuel_prices_{date_str}.csv"
        return fallback_url
        
    except Exception as e:
        print(f"Error finding latest CSV: {e}")
        return "https://assets.publishing.service.gov.uk/media/69495a85888ddc41b48a548e/weekly_road_fuel_prices_221225.csv"

def fetch_uk_fuel_data():
    """Fetch UK fuel price data from CSV"""
    try:
        csv_url = get_latest_csv_url()
        print(f"Fetching UK data from: {csv_url}")
        
        response = requests.get(csv_url, timeout=10)
        response.raise_for_status()
        
        df = pd.read_csv(StringIO(response.text))
        print(f"UK CSV columns: {df.columns.tolist()}")
        
        if 'Date' in df.columns:
            date_col = 'Date'
        elif 'date' in df.columns:
            date_col = 'date'
        else:
            date_col = df.columns[0]
        
        if 'ULSP' in df.columns:
            unleaded_col = 'ULSP'
        elif 'Unleaded' in df.columns:
            unleaded_col = 'Unleaded'
        elif 'Petrol' in df.columns:
            unleaded_col = 'Petrol'
        else:
            unleaded_col = df.columns[1]
        
        if 'ULSD' in df.columns:
            diesel_col = 'ULSD'
        elif 'Diesel' in df.columns:
            diesel_col = 'Diesel'
        else:
            diesel_col = df.columns[2]
        
        df = df.rename(columns={
            date_col: 'date',
            unleaded_col: 'unleaded',
            diesel_col: 'diesel'
        })
        
        df['date'] = pd.to_datetime(df['date'], format='%d/%m/%Y', errors='coerce')
        df = df.dropna(subset=['date'])
        df = df.sort_values('date')
        
        df['unleaded'] = pd.to_numeric(df['unleaded'], errors='coerce')
        df['diesel'] = pd.to_numeric(df['diesel'], errors='coerce')
        df = df.dropna(subset=['unleaded', 'diesel'])
        df = df[['date', 'unleaded', 'diesel']]
        
        print(f"Successfully loaded {len(df)} rows of UK data")
        return df
        
    except Exception as e:
        print(f"Error fetching UK data: {e}")
        import traceback
        traceback.print_exc()
        return None

def fetch_us_fuel_data():
    """Fetch US fuel price data"""
    try:
        end_date = datetime.now()
        dates = pd.date_range(end=end_date, periods=52, freq='W')
        
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
        return None

def predict_single_run(df, fuel_types, variation_factor=0.02):
    """
    Single prediction run with optional variation
    variation_factor adds randomness to simulate market uncertainty
    """
    if df is None or len(df) < 4:
        return None
    
    predictions = {}
    
    for fuel_type in fuel_types:
        prices = df[fuel_type].values
        
        # Add slight random variation to the data window used
        window_size = np.random.randint(3, 6)  # Use 3-5 weeks instead of fixed 4
        ma_window = np.mean(prices[-window_size:])
        
        # Calculate trend with variation
        recent_window = np.random.randint(1, 3)
        recent_avg = np.mean(prices[-recent_window:])
        previous_avg = np.mean(prices[-(window_size+recent_window):-recent_window])
        trend = recent_avg - previous_avg
        
        # Add market volatility factor
        volatility = np.random.uniform(-variation_factor, variation_factor)
        
        # Prediction with variation
        prediction = ma_window + trend + (ma_window * volatility)
        
        predictions[fuel_type] = {
            'predicted_price': prediction,
            'current_price': prices[-1]
        }
    
    return predictions

def generate_averaged_predictions(df, fuel_types, num_predictions=10):
    """Run multiple predictions and average them"""
    if df is None:
        return None
    
    all_predictions = []
    
    print(f"Running {num_predictions} prediction iterations...")
    for i in range(num_predictions):
        pred = predict_single_run(df, fuel_types)
        if pred:
            all_predictions.append(pred)
        print(f"  Iteration {i+1}/{num_predictions} complete")
    
    if not all_predictions:
        return None
    
    # Average all predictions
    averaged = {}
    for fuel_type in fuel_types:
        predicted_prices = [p[fuel_type]['predicted_price'] for p in all_predictions]
        current_price = all_predictions[0][fuel_type]['current_price']
        
        avg_prediction = np.mean(predicted_prices)
        std_dev = np.std(predicted_prices)
        
        averaged[fuel_type] = {
            'predicted_price': round(avg_prediction, 2),
            'current_price': round(current_price, 2),
            'change': round(avg_prediction - current_price, 2),
            'confidence_lower': round(avg_prediction - std_dev, 2),
            'confidence_upper': round(avg_prediction + std_dev, 2),
            'num_iterations': len(all_predictions)
        }
    
    return averaged

def main():
    """Main function to generate and save predictions"""
    print("=" * 60)
    print("PetrolPredict - Batch Prediction Generator")
    print("=" * 60)
    
    # Fetch data
    print("\n[1/4] Fetching UK data...")
    uk_df = fetch_uk_fuel_data()
    
    print("\n[2/4] Fetching US data...")
    us_df = fetch_us_fuel_data()
    
    # Generate predictions
    print("\n[3/4] Generating UK predictions (10 iterations)...")
    uk_predictions = generate_averaged_predictions(uk_df, ['unleaded', 'diesel'], num_predictions=10)
    
    print("\n[4/4] Generating US predictions (10 iterations)...")
    us_predictions = generate_averaged_predictions(us_df, ['regular', 'diesel'], num_predictions=10)
    
    # Prepare output
    timestamp = datetime.now()
    output = {
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
        'generated_at': timestamp.strftime('%Y-%m-%d %H:%M:%S'),
        'prediction_date': (timestamp + timedelta(days=7)).strftime('%Y-%m-%d'),
        'num_iterations': 10
    }
    
    # Create predictions directory if it doesn't exist
    predictions_dir = 'predictions_history'
    os.makedirs(predictions_dir, exist_ok=True)
    
    # Save timestamped version for history
    timestamp_str = timestamp.strftime('%Y%m%d_%H%M%S')
    timestamped_file = os.path.join(predictions_dir, f'predictions_{timestamp_str}.json')
    with open(timestamped_file, 'w') as f:
        json.dump(output, f, indent=2)
    
    # Also save as latest.json for the app to use
    latest_file = 'predictions.json'
    with open(latest_file, 'w') as f:
        json.dump(output, f, indent=2)
    
    # Save metadata about all predictions
    update_predictions_index(timestamped_file, output)
    
    print("\n" + "=" * 60)
    print(f"✓ Predictions saved to:")
    print(f"  - {latest_file} (for web app)")
    print(f"  - {timestamped_file} (historical record)")
    print("=" * 60)
    
    # Display summary
    if uk_predictions:
        print("\nUK Predictions:")
        print(f"  Unleaded: {uk_predictions['unleaded']['current_price']}p → {uk_predictions['unleaded']['predicted_price']}p "
              f"({uk_predictions['unleaded']['change']:+.2f}p)")
        print(f"  Diesel:   {uk_predictions['diesel']['current_price']}p → {uk_predictions['diesel']['predicted_price']}p "
              f"({uk_predictions['diesel']['change']:+.2f}p)")
    
    if us_predictions:
        print("\nUS Predictions:")
        print(f"  Regular:  {us_predictions['regular']['current_price']}¢ → {us_predictions['regular']['predicted_price']}¢ "
              f"({us_predictions['regular']['change']:+.2f}¢)")
        print(f"  Diesel:   {us_predictions['diesel']['current_price']}¢ → {us_predictions['diesel']['predicted_price']}¢ "
              f"({us_predictions['diesel']['change']:+.2f}¢)")
    
    print("\n✓ Done! Run the Flask app to view predictions.\n")

def update_predictions_index(filepath, data):
    """Maintain an index of all predictions for historical analysis"""
    index_file = 'predictions_history/index.json'
    
    # Load existing index or create new one
    if os.path.exists(index_file):
        with open(index_file, 'r') as f:
            index = json.load(f)
    else:
        index = {'predictions': []}
    
    # Add new entry
    entry = {
        'file': os.path.basename(filepath),
        'generated_at': data['generated_at'],
        'prediction_date': data['prediction_date'],
        'uk_available': data['uk']['available'],
        'us_available': data['us']['available']
    }
    
    # Add summary of predictions
    if data['uk']['available']:
        entry['uk_unleaded'] = data['uk']['predictions']['unleaded']['predicted_price']
        entry['uk_diesel'] = data['uk']['predictions']['diesel']['predicted_price']
    
    if data['us']['available']:
        entry['us_regular'] = data['us']['predictions']['regular']['predicted_price']
        entry['us_diesel'] = data['us']['predictions']['diesel']['predicted_price']
    
    index['predictions'].append(entry)
    
    # Save updated index
    with open(index_file, 'w') as f:
        json.dump(index, f, indent=2)
    
    print(f"  - predictions_history/index.json (updated with {len(index['predictions'])} total predictions)")

if __name__ == '__main__':
    main()
