from flask import Flask, render_template, request, jsonify
import joblib

app = Flask(__name__)

# Load the saved model and scaler
model = joblib.load('model.joblib')
scaler = joblib.load('scaler.joblib')

@app.route('/')
def home():
    # Render the index.html template when visiting the root route
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input data from the request
        data = request.get_json(force=True)
        features = data['features']

        # Scale the features before making predictions
        features_scaled = scaler.transform([features])

        # Make a prediction using the trained model
        prediction = model.predict(features_scaled)

        # Return the prediction as JSON
        return jsonify({'prediction': prediction[0]})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
