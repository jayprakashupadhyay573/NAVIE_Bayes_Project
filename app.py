from flask import Flask, render_template, request
import numpy as np
import pickle
import sklearn
app = Flask(__name__)

# Load trained Gaussian Naive Bayes model
with open('gnb.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Read form values
        fresh = float(request.form['Fresh'])
        milk = float(request.form['Milk'])
        grocery = float(request.form['Grocery'])
        frozen = float(request.form['Frozen'])
        detergents = float(request.form['Detergents_Paper'])
        delicassen = float(request.form['Delicassen'])

        # Same preprocessing as training
        X = np.log1p([[fresh, milk, grocery, frozen, detergents, delicassen]])

        # Model prediction
        pred = model.predict(X)[0]
        proba = model.predict_proba(X)
        confidence = np.max(proba) * 100

        # Business logic
        if pred == 1:
            customer_type = "HORECA Customer (Hotels / Restaurants / Cafes)"
            suggestion = (
                "• Focus on <b>Fresh & Frozen supply contracts</b><br>"
                "• Offer <b>bulk fresh produce discounts</b><br>"
                "• Promote <b>daily delivery & freshness assurance</b>"
            )
        else:
            customer_type = "Retail Customer (Supermarkets / Grocery Stores)"
            suggestion = (
                "• Promote <b>Grocery, Milk & Detergent bundles</b><br>"
                "• Offer <b>volume-based pricing</b><br>"
                "• Push <b>FMCG & packaged products</b>"
            )

        return render_template(
            'index.html',
            prediction=customer_type,
            confidence=f"{confidence:.2f}%",
            suggestion=suggestion
        )

    except:
        return render_template(
            'index.html',
            prediction="Invalid Input",
            confidence="—",
            suggestion="Please enter values within allowed range."
        )

import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
