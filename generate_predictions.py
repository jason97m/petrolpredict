#!/usr/bin/env python3
"""
Generate fuel price predictions
This script loads the data, trains the models, and generates predictions
Can be run standalone or imported by the Flask app
"""

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from datetime import datetime, timedelta
import numpy as np
import json
import os

DATA_DIR = "fuel_data"

def load_data():
    """Load and prepare fuel price data from multiple sources"""
    # --- UK petrol prices ---
    uk_gas = pd.read_csv(f"{DATA_DIR}/uk_petrol_weekly.csv")
    uk_gas = uk_gas.rename(columns={
        "Date": "date",
        "ULSP (Ultra low sulphur unleaded petrol) Pump price in pence/litre": "uk_petrol_price"
    })
    uk_gas["date"] = pd.to_datetime(uk_gas["date"], dayfirst=True, errors='coerce')
    uk_gas = uk_gas.dropna(subset=['date', 'uk_petrol_price']).sort_values("date")
    print(f"UK petrol date range: {uk_gas['date'].min()} to {uk_gas['date'].max()}")
    
    # --- Brent prices ---
    try:
        brent = pd.read_csv(f"{DATA_DIR}/brent_weekly.csv")
        brent.columns = ["date", "brent_price"]
        brent["date"] = pd.to_datetime(brent["date"], errors='coerce')
        brent = brent.dropna(subset=['date', 'brent_price']).sort_values("date")
        print(f"Brent date range: {brent['date'].min()} to {brent['date'].max()}")
        df = pd.merge_asof(uk_gas, brent, on="date", direction="nearest")
        df["brent_price"] = df["brent_price"].ffill()
    except Exception as e:
        print(f"Brent data not available: {e}")
        df = uk_gas.copy()
        df["brent_price"] = np.nan
    
    # --- US gasoline prices ---
    try:
        us_gas = pd.read_excel(
            f"{DATA_DIR}/us_gas_weekly.xls",
            sheet_name="Data 1",
            skiprows=2
        )
        us_gas = us_gas.iloc[:, [0, 1]]
        us_gas.columns = ['date', 'us_gas_price']
        us_gas['date'] = pd.to_datetime(us_gas['date'], errors='coerce')
        us_gas['us_gas_price'] = pd.to_numeric(us_gas['us_gas_price'], errors='coerce')
        us_gas = us_gas.dropna(subset=['date', 'us_gas_price']).sort_values("date")
        print(f"US gas date range: {us_gas['date'].min()} to {us_gas['date'].max()}")
        df = pd.merge_asof(df, us_gas, on="date", direction="nearest")
        df["us_gas_price"] = df["us_gas_price"].ffill()
    except Exception as e:
        print(f"US gas data not available: {e}")
        df["us_gas_price"] = np.nan
    
    return df

def create_features_uk(df):
    """Create enhanced features for UK price predictions"""
    # Multiple lag features
    for i in [1, 2, 4]:
        df[f"uk_gas_lag{i}"] = df["uk_petrol_price"].shift(i)
        df[f"brent_lag{i}"] = df["brent_price"].shift(i) if "brent_price" in df else np.nan
        df[f"us_gas_lag{i}"] = df["us_gas_price"].shift(i) if "us_gas_price" in df else np.nan
    
    # Rolling averages
    df["uk_gas_ma4"] = df["uk_petrol_price"].rolling(window=4, min_periods=1).mean()
    df["uk_gas_ma8"] = df["uk_petrol_price"].rolling(window=8, min_periods=1).mean()
    
    # Price changes
    df["uk_gas_change"] = df["uk_petrol_price"] - df["uk_gas_lag1"]
    df["brent_change"] = df["brent_price"] - df["brent_lag1"] if "brent_price" in df else np.nan
    
    # Time-based features
    df["week_of_year"] = df["date"].dt.isocalendar().week
    df["month"] = df["date"].dt.month
    
    # Keep only rows with sufficient data
    df = df.dropna(subset=["uk_gas_lag1"]).reset_index(drop=True)
    
    if df.empty:
        raise ValueError("Dataframe is empty after creating UK features.")
    
    return df

def create_features_us(df):
    """Create enhanced features for US price predictions"""
    # Multiple lag features
    for i in [1, 2, 4]:
        df[f"us_gas_lag{i}"] = df["us_gas_price"].shift(i)
        df[f"brent_lag{i}"] = df["brent_price"].shift(i) if "brent_price" in df else np.nan
        df[f"uk_gas_lag{i}"] = df["uk_petrol_price"].shift(i) if "uk_petrol_price" in df else np.nan
    
    # Rolling averages
    df["us_gas_ma4"] = df["us_gas_price"].rolling(window=4, min_periods=1).mean()
    df["us_gas_ma8"] = df["us_gas_price"].rolling(window=8, min_periods=1).mean()
    
    # Price changes
    df["us_gas_change"] = df["us_gas_price"] - df["us_gas_lag1"]
    df["brent_change"] = df["brent_price"] - df["brent_lag1"] if "brent_price" in df else np.nan
    
    # Time-based features
    df["week_of_year"] = df["date"].dt.isocalendar().week
    df["month"] = df["date"].dt.month
    
    # Keep only rows with sufficient data
    df = df.dropna(subset=["us_gas_lag1"]).reset_index(drop=True)
    
    if df.empty:
        raise ValueError("Dataframe is empty after creating US features.")
    
    return df

def train_models(df_uk, df_us):
    """Train prediction models for both UK and US"""
    
    # UK Model Training
    feature_cols_uk = [
        "brent_lag1", "us_gas_lag1", "uk_gas_lag1",
        "brent_lag2", "us_gas_lag2", "uk_gas_lag2",
        "uk_gas_ma4", "uk_gas_change", "brent_change",
        "week_of_year", "month"
    ]
    
    X_uk = df_uk[feature_cols_uk].copy()
    y_uk = df_uk["uk_petrol_price"]
    X_uk = X_uk.fillna(method="ffill").fillna(0)
    
    scaler_uk = StandardScaler()
    X_uk_scaled = scaler_uk.fit_transform(X_uk)
    
    model_uk_rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
    model_uk_rf.fit(X_uk_scaled, y_uk)
    
    model_uk_lr = LinearRegression()
    model_uk_lr.fit(X_uk_scaled, y_uk)
    
    # US Model Training
    feature_cols_us = [
        "brent_lag1", "uk_gas_lag1", "us_gas_lag1",
        "brent_lag2", "uk_gas_lag2", "us_gas_lag2",
        "us_gas_ma4", "us_gas_change", "brent_change",
        "week_of_year", "month"
    ]
    
    X_us = df_us[feature_cols_us].copy()
    y_us = df_us["us_gas_price"]
    X_us = X_us.fillna(method="ffill").fillna(0)
    
    scaler_us = StandardScaler()
    X_us_scaled = scaler_us.fit_transform(X_us)
    
    model_us_rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
    model_us_rf.fit(X_us_scaled, y_us)
    
    model_us_lr = LinearRegression()
    model_us_lr.fit(X_us_scaled, y_us)
    
    return {
        'uk': {
            'rf': model_uk_rf,
            'lr': model_uk_lr,
            'scaler': scaler_uk,
            'features': feature_cols_uk,
            'df': df_uk
        },
        'us': {
            'rf': model_us_rf,
            'lr': model_us_lr,
            'scaler': scaler_us,
            'features': feature_cols_us,
            'df': df_us
        }
    }

def predict_price(models, region, target_date):
    """Generate prediction for a specific region and date"""
    
    model_data = models[region]
    df = model_data['df']
    last_row = df.iloc[-1]
    
    target_date = pd.to_datetime(target_date)
    
    if region == 'uk':
        features = {
            "brent_lag1": last_row.get("brent_price", 0),
            "us_gas_lag1": last_row.get("us_gas_price", 0),
            "uk_gas_lag1": last_row["uk_petrol_price"],
            "brent_lag2": df.iloc[-2].get("brent_price", 0) if len(df) > 1 else 0,
            "us_gas_lag2": df.iloc[-2].get("us_gas_price", 0) if len(df) > 1 else 0,
            "uk_gas_lag2": df.iloc[-2]["uk_petrol_price"] if len(df) > 1 else last_row["uk_petrol_price"],
            "uk_gas_ma4": last_row.get("uk_gas_ma4", last_row["uk_petrol_price"]),
            "uk_gas_change": last_row.get("uk_gas_change", 0),
            "brent_change": last_row.get("brent_change", 0),
            "week_of_year": target_date.isocalendar()[1],
            "month": target_date.month
        }
    else:  # US
        features = {
            "brent_lag1": last_row.get("brent_price", 0),
            "uk_gas_lag1": last_row.get("uk_petrol_price", 0),
            "us_gas_lag1": last_row["us_gas_price"],
            "brent_lag2": df.iloc[-2].get("brent_price", 0) if len(df) > 1 else 0,
            "uk_gas_lag2": df.iloc[-2].get("uk_petrol_price", 0) if len(df) > 1 else 0,
            "us_gas_lag2": df.iloc[-2]["us_gas_price"] if len(df) > 1 else last_row["us_gas_price"],
            "us_gas_ma4": last_row.get("us_gas_ma4", last_row["us_gas_price"]),
            "us_gas_change": last_row.get("us_gas_change", 0),
            "brent_change": last_row.get("brent_change", 0),
            "week_of_year": target_date.isocalendar()[1],
            "month": target_date.month
        }
    
    feature_vector = [[features[col] for col in model_data['features']]]
    feature_vector_scaled = model_data['scaler'].transform(feature_vector)
    
    rf_prediction = model_data['rf'].predict(feature_vector_scaled)[0]
    lr_prediction = model_data['lr'].predict(feature_vector_scaled)[0]
    
    # Ensemble prediction
    prediction = 0.7 * rf_prediction + 0.3 * lr_prediction
    
    return {
        'date': target_date.strftime('%Y-%m-%d'),
        'prediction': round(prediction, 2),
        'rf_prediction': round(rf_prediction, 2),
        'lr_prediction': round(lr_prediction, 2),
        'current_price': round(last_row[f"{region}_{'petrol' if region == 'uk' else 'gas'}_price"], 2)
    }

def main():
    """Main function to generate predictions"""
    print("="*60)
    print("Generating Fuel Price Predictions")
    print("="*60)
    
    # Load data
    print("\nLoading data...")
    df = load_data()
    
    # Create features
    print("\nCreating features...")
    df_uk = create_features_uk(df.copy())
    df_us = create_features_us(df.copy())
    
    # Train models
    print("\nTraining models...")
    models = train_models(df_uk, df_us)
    
    # Generate predictions for next week
    target_date = datetime(2026, 1, 12)  # Adjust as needed
    
    print("\nGenerating predictions...")
    uk_prediction = predict_price(models, 'uk', target_date)
    us_prediction = predict_price(models, 'us', target_date)
    
    # Display results
    print("\n" + "="*60)
    print("PREDICTIONS FOR", target_date.strftime('%B %d, %Y'))
    print("="*60)
    
    print("\n🇬🇧 UK Petrol Price:")
    print(f"  Current:    {uk_prediction['current_price']} pence/liter")
    print(f"  Predicted:  {uk_prediction['prediction']} pence/liter")
    print(f"  Change:     {uk_prediction['prediction'] - uk_prediction['current_price']:+.2f} pence/liter")
    print(f"  RF Model:   {uk_prediction['rf_prediction']} pence/liter")
    print(f"  LR Model:   {uk_prediction['lr_prediction']} pence/liter")
    
    print("\n🇺🇸 US Gasoline Price:")
    print(f"  Current:    ${us_prediction['current_price']}/gallon")
    print(f"  Predicted:  ${us_prediction['prediction']}/gallon")
    print(f"  Change:     ${us_prediction['prediction'] - us_prediction['current_price']:+.2f}/gallon")
    print(f"  RF Model:   ${us_prediction['rf_prediction']}/gallon")
    print(f"  LR Model:   ${us_prediction['lr_prediction']}/gallon")
    
    # Save predictions to JSON
    output = {
        'generated_at': datetime.now().isoformat(),
        'prediction_date': target_date.strftime('%Y-%m-%d'),
        'uk': uk_prediction,
        'us': us_prediction
    }
    
    output_file = 'predictions.json'
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\n✓ Predictions saved to {output_file}")
    print("="*60)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n✗ Error generating predictions: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
