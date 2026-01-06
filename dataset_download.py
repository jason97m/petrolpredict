import os
import requests
import pandas as pd

# ------------------------------------------------------------------
# Setup
# ------------------------------------------------------------------
DATA_DIR = "fuel_data"
os.makedirs(DATA_DIR, exist_ok=True)

def download_file(url, filename):
    """Download a file from a URL (binary)"""
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    path = os.path.join(DATA_DIR, filename)
    with open(path, "wb") as f:
        f.write(response.content)
    print(f"Downloaded {filename}")
    return path

# ------------------------------------------------------------------
# 1. US Gasoline Weekly Prices (XLS -> CSV)
# ------------------------------------------------------------------
US_GAS_XLS_URL = "https://www.eia.gov/petroleum/gasdiesel/xls/pswrgvwall.xls"
us_gas_xls_path = download_file(US_GAS_XLS_URL, "us_gas_weekly.xls")

# Read the first sheet
df_us_gas = pd.read_excel(us_gas_xls_path, sheet_name=0)
us_gas_csv = os.path.join(DATA_DIR, "us_gasoline_weekly.csv")
df_us_gas.to_csv(us_gas_csv, index=False)
print(f"US gasoline weekly CSV saved to: {us_gas_csv}")

# ------------------------------------------------------------------
# 2. Weekly crude oil prices (CSV)
# ------------------------------------------------------------------
CRUDE_BRENT_WEEKLY_URL = "https://datahub.io/@olayway/oil-prices/_r/-/data/brent-weekly.csv"
CRUDE_WTI_WEEKLY_URL = "https://datahub.io/@olayway/oil-prices/_r/-/data/wti-weekly.csv"

brent_csv_path = download_file(CRUDE_BRENT_WEEKLY_URL, "brent_weekly.csv")
wti_csv_path   = download_file(CRUDE_WTI_WEEKLY_URL, "wti_weekly.csv")

# ------------------------------------------------------------------
# 3. UK Petrol Weekly Prices (CSV)
# ------------------------------------------------------------------
UK_PETROL_WEEKLY_CSV_URL = "https://assets.publishing.service.gov.uk/media/69495a85888ddc41b48a548e/weekly_road_fuel_prices_221225.csv"
uk_petrol_csv_path = download_file(UK_PETROL_WEEKLY_CSV_URL, "uk_petrol_weekly.csv")

# Read and preview the UK petrol CSV
df_uk_petrol = pd.read_csv(uk_petrol_csv_path)
print("\nUK petrol weekly head:\n", df_uk_petrol.head())

print("\nAll datasets downloaded and saved:")
print(f"- {us_gas_csv}")
print(f"- {brent_csv_path}")
print(f"- {wti_csv_path}")
print(f"- {uk_petrol_csv_path}")
