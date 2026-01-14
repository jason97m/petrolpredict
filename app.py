from flask import Flask, render_template, jsonify
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from datetime import datetime, timedelta
import numpy as np

app = Flask(__name__)
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

# Load and prepare data
df = load_data()

# Create separate feature sets for UK and US
df_uk = create_features_uk(df.copy())
df_us = create_features_us(df.copy())

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

def predict_uk(target_date=None):
    """Predict UK fuel price for a specific week"""
    if target_date is None:
        last_row = df_uk.iloc[-1]
        target_date = last_row["date"] + timedelta(weeks=1)
    else:
        target_date = pd.to_datetime(target_date)
    
    last_row = df_uk.iloc[-1]
    
    features = {
        "brent_lag1": last_row.get("brent_price", 0),
        "us_gas_lag1": last_row.get("us_gas_price", 0),
        "uk_gas_lag1": last_row["uk_petrol_price"],
        "brent_lag2": df_uk.iloc[-2].get("brent_price", 0) if len(df_uk) > 1 else 0,
        "us_gas_lag2": df_uk.iloc[-2].get("us_gas_price", 0) if len(df_uk) > 1 else 0,
        "uk_gas_lag2": df_uk.iloc[-2]["uk_petrol_price"] if len(df_uk) > 1 else last_row["uk_petrol_price"],
        "uk_gas_ma4": last_row.get("uk_gas_ma4", last_row["uk_petrol_price"]),
        "uk_gas_change": last_row.get("uk_gas_change", 0),
        "brent_change": last_row.get("brent_change", 0),
        "week_of_year": target_date.isocalendar()[1],
        "month": target_date.month
    }
    
    feature_vector = [[features[col] for col in feature_cols_uk]]
    feature_vector_scaled = scaler_uk.transform(feature_vector)
    
    rf_prediction = model_uk_rf.predict(feature_vector_scaled)[0]
    lr_prediction = model_uk_lr.predict(feature_vector_scaled)[0]
    prediction = 0.7 * rf_prediction + 0.3 * lr_prediction
    
    return target_date, round(prediction, 2), round(rf_prediction, 2), round(lr_prediction, 2)

def predict_us(target_date=None):
    """Predict US fuel price for a specific week"""
    if target_date is None:
        last_row = df_us.iloc[-1]
        target_date = last_row["date"] + timedelta(weeks=1)
    else:
        target_date = pd.to_datetime(target_date)
    
    last_row = df_us.iloc[-1]
    
    features = {
        "brent_lag1": last_row.get("brent_price", 0),
        "uk_gas_lag1": last_row.get("uk_petrol_price", 0),
        "us_gas_lag1": last_row["us_gas_price"],
        "brent_lag2": df_us.iloc[-2].get("brent_price", 0) if len(df_us) > 1 else 0,
        "uk_gas_lag2": df_us.iloc[-2].get("uk_petrol_price", 0) if len(df_us) > 1 else 0,
        "us_gas_lag2": df_us.iloc[-2]["us_gas_price"] if len(df_us) > 1 else last_row["us_gas_price"],
        "us_gas_ma4": last_row.get("us_gas_ma4", last_row["us_gas_price"]),
        "us_gas_change": last_row.get("us_gas_change", 0),
        "brent_change": last_row.get("brent_change", 0),
        "week_of_year": target_date.isocalendar()[1],
        "month": target_date.month
    }
    
    feature_vector = [[features[col] for col in feature_cols_us]]
    feature_vector_scaled = scaler_us.transform(feature_vector)
    
    rf_prediction = model_us_rf.predict(feature_vector_scaled)[0]
    lr_prediction = model_us_lr.predict(feature_vector_scaled)[0]
    prediction = 0.7 * rf_prediction + 0.3 * lr_prediction
    
    return target_date, round(prediction, 2), round(rf_prediction, 2), round(lr_prediction, 2)

def get_statistics():
    """Calculate helpful statistics for both regions"""
    recent_uk = df_uk.tail(52)
    recent_us = df_us.tail(52)
    
    return {
        "uk": {
            "current_price": round(df_uk.iloc[-1]["uk_petrol_price"], 2),
            "avg_year": round(recent_uk["uk_petrol_price"].mean(), 2),
            "min_year": round(recent_uk["uk_petrol_price"].min(), 2),
            "max_year": round(recent_uk["uk_petrol_price"].max(), 2),
            "last_update": df_uk.iloc[-1]["date"].strftime("%Y-%m-%d")
        },
        "us": {
            "current_price": round(df_us.iloc[-1]["us_gas_price"], 2),
            "avg_year": round(recent_us["us_gas_price"].mean(), 2),
            "min_year": round(recent_us["us_gas_price"].min(), 2),
            "max_year": round(recent_us["us_gas_price"].max(), 2),
            "last_update": df_us.iloc[-1]["date"].strftime("%Y-%m-%d")
        }
    }

@app.route("/")
def index():
    """Main dashboard route"""
    # Get recent data for chart (6 months)
    recent_df_uk = df_uk.tail(26).copy()
    recent_df_us = df_us.tail(26).copy()
    
    # Predict for the following sunday
    # target_date = datetime(2026, 1, 12)
    target_date = datetime.now() + timedelta(days=(6 - datetime.now().weekday()) % 7 or 7)
    
    # UK Prediction
    uk_pred_date, uk_pred_price, uk_rf_pred, uk_lr_pred = predict_uk(target_date)
    
    # US Prediction
    us_pred_date, us_pred_price, us_rf_pred, us_lr_pred = predict_us(target_date)
    
    # Prepare chart data for UK
    plot_df_uk = recent_df_uk[["date", "uk_petrol_price", "us_gas_price", "brent_price"]].copy()
    plot_df_uk = pd.concat([
        plot_df_uk,
        pd.DataFrame([{
            "date": uk_pred_date,
            "uk_petrol_price": uk_pred_price,
            "us_gas_price": np.nan,
            "brent_price": np.nan
        }])
    ], ignore_index=True)
    
    # Prepare chart data for US
    plot_df_us = recent_df_us[["date", "us_gas_price", "uk_petrol_price", "brent_price"]].copy()
    plot_df_us = pd.concat([
        plot_df_us,
        pd.DataFrame([{
            "date": us_pred_date,
            "us_gas_price": us_pred_price,
            "uk_petrol_price": np.nan,
            "brent_price": np.nan
        }])
    ], ignore_index=True)
    
    data_uk = {
        "date": plot_df_uk["date"].dt.strftime("%Y-%m-%d").tolist(),
        "uk": plot_df_uk["uk_petrol_price"].tolist(),
        "us": plot_df_uk["us_gas_price"].tolist(),
        "brent": plot_df_uk["brent_price"].tolist()
    }
    
    data_us = {
        "date": plot_df_us["date"].dt.strftime("%Y-%m-%d").tolist(),
        "us": plot_df_us["us_gas_price"].tolist(),
        "uk": plot_df_us["uk_petrol_price"].tolist(),
        "brent": plot_df_us["brent_price"].tolist()
    }
    
    stats = get_statistics()
    
    return render_template(
        "index.html",
        data_uk=data_uk,
        data_us=data_us,
        uk_pred_price=uk_pred_price,
        us_pred_price=us_pred_price,
        pred_date=uk_pred_date.strftime("%B %d, %Y"),
        stats=stats,
        uk_rf_pred=uk_rf_pred,
        uk_lr_pred=uk_lr_pred,
        us_rf_pred=us_rf_pred,
        us_lr_pred=us_lr_pred
    )

@app.route("/api/predict")
def api_predict():
    """API endpoint for predictions"""
    target_date = datetime(2026, 1, 12)
    
    uk_pred_date, uk_pred_price, uk_rf_pred, uk_lr_pred = predict_uk(target_date)
    us_pred_date, us_pred_price, us_rf_pred, us_lr_pred = predict_us(target_date)
    
    return jsonify({
        "prediction_date": uk_pred_date.strftime("%Y-%m-%d"),
        "uk": {
            "predicted_price": uk_pred_price,
            "rf_prediction": uk_rf_pred,
            "lr_prediction": uk_lr_pred,
            "current_price": round(df_uk.iloc[-1]["uk_petrol_price"], 2),
            "unit": "pence/liter"
        },
        "us": {
            "predicted_price": us_pred_price,
            "rf_prediction": us_rf_pred,
            "lr_prediction": us_lr_pred,
            "current_price": round(df_us.iloc[-1]["us_gas_price"], 2),
            "unit": "$/gallon"
        }
    })

if __name__ == "__main__":
    app.run(debug=True)
