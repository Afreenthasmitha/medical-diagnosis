from flask import Flask, request, jsonify, render_template
import numpy as np
import pickle

app = Flask(__name__)
model = pickle.load(open('medical_model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    features = [float(x) for x in request.form.values()]
    final_input = np.array([features])
    prediction = model.predict(final_input)[0]
    result = "🩸 High Risk of Diabetes" if prediction == 1 else "✅ Low Risk of Diabetes"
    return render_template('index.html', prediction_text=result)

if __name__ == "__main__":
    app.run(debug=True)
