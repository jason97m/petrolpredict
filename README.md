0~# PetrolPredict

A Flask web application that predicts UK fuel prices for the upcoming week based on historical data.

## Features

- Fetches UK fuel price data
- Predicts next week's prices for Unleaded and Diesel
- Clean, responsive web interface
- RESTful API endpoint for predictions

## Project Structure

```
petrolpredict/
│
├── app.py                 # Main Flask application
├── requirements.txt       # Python dependencies
├── Procfile              # Heroku deployment config
├── runtime.txt           # Python version for Heroku
├── .gitignore            # Git ignore rules
├── README.md             # This file
│
└── templates/
    └── index.html        # Main webpage template
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

5. **Run the application**
   ```bash
   python app.py
   ```
   
   The app will be available at `http://localhost:5000`

## Deployment to Heroku

### Prerequisites

- Heroku account (free tier available)
- Heroku CLI installed
- Git initialized in your project

### Deployment Steps

1. **Login to Heroku**
   ```bash
   heroku login
   ```

2. **Create a new Heroku app**
   ```bash
   heroku create petrolpredict
   ```
   (Replace `petrolpredict` with your preferred app name if it's taken)

3. **Initialize Git (if not already done)**
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   ```

4. **Push to Heroku**
   ```bash
   git push heroku main
   ```
   
   If your default branch is `master`:
   ```bash
   git push heroku master
   ```

5. **Open your app**
   ```bash
   heroku open
   ```

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
Returns the main web interface

### GET /api/predictions
Returns JSON with fuel price predictions

**Response format:**
```json
{
  "predictions": {
    "unleaded": {
      "predicted_price": 155.23,
      "current_price": 154.50,
      "change": 0.73,
      "confidence_lower": 152.10,
      "confidence_upper": 158.36
    },
    "diesel": {
      "predicted_price": 159.87,
      "current_price": 158.90,
      "change": 0.97,
      "confidence_lower": 156.45,
      "confidence_upper": 163.29
    }
  },
  "last_updated": "2025-12-31 10:30",
  "prediction_date": "2026-01-07"
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
