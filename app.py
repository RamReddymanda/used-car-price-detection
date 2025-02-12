from flask import Flask, render_template, request
import pickle
import pandas as pd
import numpy as np
import tensorflow as tf

app = Flask(__name__)

# Load models and encoders
with open('models/model.pkl', 'rb') as file:
    model = pickle.load(file)
scaler = pickle.load(open("models/scaler.pkl", 'rb'))




encoder = pickle.load(open("models/encoder.pkl", 'rb'))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    # Get form data
    Km = request.form.get('km')
    age = request.form.get('age')
    make = request.form.get('make')
    models = request.form.get('model')
    drivetrain = request.form.get('drivetrain')
    transmission = request.form.get('transmission')
    state = request.form.get('state')
    city = request.form.get('city')
    trim = request.form.get('trim')
    bodytype = request.form.get('body_type')
    vehicletype = request.form.get('vehicle_type')
    price = request.form.get('price')

    # Create DataFrame
    my_car = [Km, age, make, models, drivetrain, transmission, state, city, trim, bodytype, vehicletype, price]
    sample = pd.DataFrame([my_car], columns=['miles', 'age', 'make', 'model', 'drivetrain', 'transmission', 'state', 'city', 'trim', 'body_type', 'vehicle_type', 'price'])

    # Transform features
    feature_num = scaler.transform(sample[['miles', 'age']])
    feature_cat = encoder.transform(sample[['make', 'model', 'drivetrain', 'transmission', 'state', 'city', 'trim', 'body_type', 'vehicle_type']])

    # Concatenate features
    features = np.concatenate([feature_num, feature_cat], axis=1)

    # Debug print statements
    print("Numerical features:", feature_num)
    print("Categorical features:", feature_cat)
    print("Concatenated features:", features)

    # Make prediction
    prediction = model.predict(features)
    print("Prediction:", prediction)

    return render_template("index.html", prediction=prediction[0][0] if prediction else "No prediction")

if __name__ == "__main__":
    app.run(debug=True
