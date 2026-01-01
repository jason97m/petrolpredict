from flask import Flask, render_template, jsonify
import json
import os
from datetime import datetime

app = Flask(__name__)

PREDICTIONS_FILE = 'predictions.json'
HISTORY_DIR = 'predictions_history'
INDEX_FILE = os.path.join(HISTORY_DIR, 'index.json')

def load_predictions():
    """Load pre-generated predictions from JSON file"""
    try:
        if not os.path.exists(PREDICTIONS_FILE):
            return {
                'error': 'No predictions available. Run generate_predictions.py first.',
                'status': 'missing_file'
            }
        
        with open(PREDICTIONS_FILE, 'r') as f:
            data = json.load(f)
        
        # Check if predictions are stale (older than 7 days)
        generated_at = datetime.strptime(data['generated_at'], '%Y-%m-%d %H:%M:%S')
        age_days = (datetime.now() - generated_at).days
        
        if age_days > 7:
            data['warning'] = f'Predictions are {age_days} days old. Consider regenerating.'
        
        return data
        
    except Exception as e:
        print(f"Error loading predictions: {e}")
        return {
            'error': f'Error loading predictions: {str(e)}',
            'status': 'error'
        }

def load_history_index():
    """Load the index of all historical predictions"""
    try:
        if not os.path.exists(INDEX_FILE):
            return {'predictions': []}
        
        with open(INDEX_FILE, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading history index: {e}")
        return {'predictions': []}

@app.route('/')
def index():
    """Main page showing predictions"""
    return render_template('index.html')

@app.route('/history')
def history():
    """Historical predictions page"""
    return render_template('history.html')

@app.route('/api/predictions')
def get_predictions():
    """API endpoint for predictions"""
    data = load_predictions()
    
    if 'error' in data:
        return jsonify(data), 500
    
    # Add last_updated for compatibility
    data['last_updated'] = data['generated_at']
    
    return jsonify(data)

@app.route('/api/history')
def get_history():
    """API endpoint for historical predictions"""
    index = load_history_index()
    
    # Sort by date, newest first
    if index['predictions']:
        index['predictions'].sort(key=lambda x: x['generated_at'], reverse=True)
    
    return jsonify(index)

@app.route('/api/history/<filename>')
def get_historical_prediction(filename):
    """Get a specific historical prediction file"""
    try:
        filepath = os.path.join(HISTORY_DIR, filename)
        
        if not os.path.exists(filepath):
            return jsonify({'error': 'File not found'}), 404
        
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        return jsonify(data)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/status')
def get_status():
    """Check prediction file status"""
    if not os.path.exists(PREDICTIONS_FILE):
        return jsonify({
            'status': 'missing',
            'message': 'No predictions file found. Run generate_predictions.py'
        })
    
    try:
        with open(PREDICTIONS_FILE, 'r') as f:
            data = json.load(f)
        
        generated_at = datetime.strptime(data['generated_at'], '%Y-%m-%d %H:%M:%S')
        age_days = (datetime.now() - generated_at).days
        
        return jsonify({
            'status': 'ok',
            'generated_at': data['generated_at'],
            'age_days': age_days,
            'prediction_date': data['prediction_date'],
            'stale': age_days > 7
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        })

if __name__ == '__main__':
    app.run(debug=True)
