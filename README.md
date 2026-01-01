### GET /api/history
Returns index of all historical predictions

**Response format:**
```json
{
  "predictions": [
    {
      "file": "predictions_20251231_103045.json",
      "generated_at": "2025-12-31 10:30:45",
      "prediction_date": "2026-01-07",
      "uk_available": true,
      "us_available": true,
      "uk_unleaded": 155.23,
      "uk_diesel": 159.87,
      "us_regular": 352.45,
      "us_diesel": 381.20
    },
    ...
  ]
}
```

### GET /api/history/<filename>
Returns a specific historical prediction file# PetrolPredict

A Flask web application that predicts UK fuel prices for the upcoming week based on historical data.

## How It Works

PetrolPredict uses a two-step process with historical tracking:

1. **Prediction Generation** (`generate_predictions.py`) - Runs 10 prediction iterations with slight variations and averages the results for more accurate forecasts. Saves results to:
   - `predictions.json` - Latest predictions (used by web app)
   - `predictions_history/predictions_YYYYMMDD_HHMMSS.json` - Timestamped historical record
   - `predictions_history/index.json` - Index of all predictions for quick lookup

2. **Web Server** (`app.py`) - Serves predictions via two pages:
   - `/` - Current predictions
   - `/history` - Historical predictions with statistics

This separation allows for:
- More accurate predictions (averaged from 10 runs)
- Faster page loads (no calculation at runtime)
- Historical tracking and analysis
- Scheduled updates (generate new predictions weekly)
- Better resource management

## Project Structure

```
petrolpredict/
│
├── app.py                      # Main Flask application (serves predictions)
├── generate_predictions.py     # Batch prediction generator (run separately)
├── schedule_predictions.py     # Optional scheduler for automatic updates
├── predictions.json            # Latest predictions (created by generate_predictions.py)
├── requirements.txt            # Python dependencies
├── Procfile                    # Heroku deployment config
├── runtime.txt                 # Python version for Heroku
├── .gitignore                  # Git ignore rules
├── README.md                   # This file
│
├── predictions_history/        # Historical predictions directory
│   ├── index.json             # Index of all predictions
│   ├── predictions_20251231_103045.json
│   ├── predictions_20260107_062315.json
│   └── ...                    # More timestamped predictions
│
└── templates/
    ├── index.html             # Main webpage template
    └── history.html           # Historical predictions page
```

## Local Development Setup

### Prerequisites

- Python 3.11+
- Git
- Virtual environment tool (venv)

### Installation Steps

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/petrolpredict.git
   cd petrolpredict
   ```

2. **Create and activate virtual environment**
   
   On Windows:
   ```bash
   python -m venv venv
   venv\Scripts\activate
   ```
   
   On Mac/Linux:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Create templates folder**
   ```bash
   mkdir templates
   ```
   Then place `index.html` in the `templates` folder.

5. **Generate initial predictions**
   ```bash
   python generate_predictions.py
   ```
   
   This creates `predictions.json` with averaged predictions from 10 iterations.

6. **Run the application**
   ```bash
   python app.py
   ```
   
   The app will be available at `http://localhost:5000`

## Usage

### Generating Predictions

**Manual Generation:**
```bash
python generate_predictions.py
```

This will:
- Fetch the latest UK and US fuel price data
- Run 10 prediction iterations with slight variations
- Average the results for more accurate forecasts
- Save to `predictions.json` (latest)
- Save to `predictions_history/predictions_YYYYMMDD_HHMMSS.json` (timestamped)
- Update `predictions_history/index.json` (historical index)

**Automatic Scheduling (Optional):**
```bash
python schedule_predictions.py
```

This runs the prediction generator automatically every Monday at 6:00 AM. You can customize the schedule in `schedule_predictions.py`.

**Alternative: Use Cron (Linux/Mac) or Task Scheduler (Windows)**

Linux/Mac cron example (run every Monday at 6 AM):
```bash
0 6 * * 1 cd /path/to/petrolpredict && /path/to/python generate_predictions.py
```

Windows Task Scheduler:
- Create a new task
- Trigger: Weekly, Monday, 6:00 AM
- Action: Run `python.exe` with argument `generate_predictions.py`
- Start in: Your petrolpredict directory

## Deployment to Heroku

### Prerequisites

- Heroku account (free tier available)
- Heroku CLI installed
- Git initialized in your project
- **IMPORTANT:** Generate predictions before deploying

### Deployment Steps

1. **Generate initial predictions locally**
   ```bash
   python generate_predictions.py
   ```
   This creates `predictions.json` which must be committed to git.

2. **Update .gitignore to include predictions.json**
   Remove `*.json` from .gitignore if present, so predictions.json is tracked.

3. **Login to Heroku**
   ```bash
   heroku login
   ```

4. **Create a new Heroku app**
   ```bash
   heroku create petrolpredict
   ```
   (Replace `petrolpredict` with your preferred app name if it's taken)

5. **Commit predictions file**
   ```bash
   git add predictions.json
   git commit -m "Add initial predictions"
   ```

6. **Push to Heroku**
   ```bash
   git push heroku main
   ```

7. **Set up automatic prediction updates (Heroku Scheduler add-on)**
   ```bash
   heroku addons:create scheduler:standard
   heroku addons:open scheduler
   ```
   
   In the scheduler dashboard:
   - Add a new job
   - Command: `python generate_predictions.py && git add predictions.json && git commit -m "Update predictions" && git push heroku main`
   - Frequency: Weekly (Monday morning recommended)
   - **Note:** This approach has limitations; see alternatives below.

8. **Open your app**
   ```bash
   heroku open
   ```

### Better Heroku Deployment Option

Since Heroku's ephemeral filesystem doesn't persist `predictions.json` between deploys, consider:

**Option A: Use a database**
- Store predictions in Heroku Postgres instead of JSON file
- Modify `generate_predictions.py` to write to database
- Modify `app.py` to read from database

**Option B: Use cloud storage**
- Store `predictions.json` in AWS S3, Google Cloud Storage, or similar
- Update both scripts to read/write from cloud storage

**Option C: GitHub Actions**
- Set up a GitHub Action to run `generate_predictions.py` weekly
- Automatically commit and push updated `predictions.json`
- Heroku auto-deploys from GitHub

## GitHub Setup

1. **Create a new repository on GitHub** (named `petrolpredict`)

2. **Connect local repository to GitHub**
   ```bash
   git remote add origin https://github.com/yourusername/petrolpredict.git
   git branch -M main
   git push -u origin main
   ```

3. **For automatic Heroku deployment from GitHub:**
   - Go to your Heroku dashboard
   - Select your app
   - Go to "Deploy" tab
   - Connect to GitHub
   - Enable automatic deploys from your main branch

## API Endpoints

### GET /
Returns the main web interface with current predictions

### GET /history
Returns the historical predictions page with statistics and timeline

### GET /api/predictions
Returns JSON with fuel price predictions (loaded from predictions.json)

**Response format:**
```json
{
  "uk": {
    "predictions": {
      "unleaded": {
        "predicted_price": 155.23,
        "current_price": 154.50,
        "change": 0.73,
        "confidence_lower": 152.10,
        "confidence_upper": 158.36,
        "num_iterations": 10
      },
      "diesel": { ... }
    },
    "currency": "pence per litre",
    "available": true
  },
  "us": { ... },
  "generated_at": "2025-12-31 10:30:00",
  "prediction_date": "2026-01-07",
  "num_iterations": 10
}
```

### GET /api/status
Returns status of prediction file

**Response format:**
```json
{
  "status": "ok",
  "generated_at": "2025-12-31 10:30:00",
  "age_days": 0,
  "prediction_date": "2026-01-07",
  "stale": false
}
```

## Customization

### Updating Data Source

The current implementation uses sample data. To use real UK government fuel price data:

1. Find the CSV URL from [gov.uk weekly road fuel prices](https://www.gov.uk/government/statistics/weekly-road-fuel-prices)
2. Update the `fetch_fuel_data()` function in `app.py` to parse the actual CSV

### Improving Predictions

The current prediction model uses a simple moving average with trend analysis. You can improve it by:

- Implementing more sophisticated time series models (ARIMA, Prophet)
- Adding seasonal adjustments
- Incorporating external factors (oil prices, exchange rates)

## Troubleshooting

### Local Development Issues

- **Port already in use**: Change the port in `app.py` or kill the process using port 5000
- **Module not found**: Ensure virtual environment is activated and dependencies are installed

### Heroku Deployment Issues

- **Application error**: Check logs with `heroku logs --tail`
- **Build failed**: Ensure all files are committed and `requirements.txt` is correct
- **App not responding**: Check dyno status with `heroku ps`

## License

MIT License - feel free to use and modify as needed.

## Contributing

Pull requests are welcome! For major changes, please open an issue first to discuss proposed changes.

## Support

For issues or questions, please open an issue on GitHub.
