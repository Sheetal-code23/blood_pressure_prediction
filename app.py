from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)
model = joblib.load('app/final_model.pkl')
scaler = joblib.load('app/scaler.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        input_features = [float(request.form[key]) for key in ['Age', 'Gender','Weight','BMI', 'Smoking', 'ExerciseHours', 'StressLevel']]
        input_scaled = scaler.transform([input_features])
        prediction = model.predict(input_scaled)[0]
        return render_template('index.html', prediction_text=f'Predicted Systolic BP: {prediction:.2f}')
    except Exception as e:
        return render_template('index.html', prediction_text=f'Error: {str(e)}')

if __name__ == '__main__':
    print("Flask app is starting..")
    app.run(debug=True)


