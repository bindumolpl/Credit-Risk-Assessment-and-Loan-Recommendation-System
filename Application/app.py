import os
from flask import Flask, request, render_template
import numpy as np
import pickle

# Paths
BASE_DIR = os.path.dirname(__file__)
MODEL_DIR = os.path.join(BASE_DIR, "Model")

app = Flask(__name__)

# Load model artifacts
with open(os.path.join(MODEL_DIR, "lending_club_pipeline.pkl"), "rb") as file:
    artifacts = pickle.load(file)

clf = artifacts['classifier']
reg = artifacts['regressor']
scaler = artifacts['scaler']
encoder = artifacts.get('encoder')
state_mapping = artifacts.get('state_mapping', {})

# Encode state
def get_encoded_state(state_val):
    state_val = state_val.upper().strip()
    if encoder:
        try:
            return encoder.transform([state_val])[0]
        except:
            return 0
    return state_mapping.get(state_val, 0)

# Home page
@app.route('/')
def index():
    return render_template('index.html')


# Prediction
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input
        req_amount = float(request.form.get('amount', 0))
        fico = float(request.form.get('risk_score', 0))
        dti = float(request.form.get('dti', 0))
        state_raw = request.form.get('state', 'NY')

        state_enc = get_encoded_state(state_raw)
        emp_len = 5  # fixed value

        # Classification input
        clf_input = scaler.transform(
            np.array([[req_amount, fico, dti, state_enc, emp_len]])
        )

        # Prediction
        is_accepted = clf.predict(clf_input)[0]

        # Confidence (correct way)
        probs = clf.predict_proba(clf_input)[0]
        confidence = round(max(probs) * 100, 2)

        # Regression (loan recommendation)
        reg_input = np.array([[fico, dti, state_enc, emp_len]])
        predicted_limit = reg.predict(reg_input)[0]

        # Decision logic
        if is_accepted == 0:
            status = "REJECTED"
            final_limit = min(predicted_limit, req_amount * 0.85)
            color = "#e74c3c"
            if confidence > 80:
                risk_level = "HIGH RISK"
            else:
                risk_level = "MEDIUM RISK"
        else:
            status = "APPROVED"
            final_limit = max(predicted_limit, req_amount)
            color = "#2ecc71"
            if confidence > 80:
                risk_level = "LOW RISK"
            else:
                risk_level = "MEDIUM RISK"       

        return render_template(
            'index.html',
            result=status,
            amount_req=f"${req_amount:,.2f}",
            counter=round(final_limit, 2),
            confidence=confidence,
            risk_level=risk_level,   # 👈 ADD THIS
            amount=req_amount,
            fico=fico,
            dti=dti,
            state=state_raw,
            color=color
        )

    except Exception as e:
        return f"Error: {str(e)}"


# Run app
if __name__ == '__main__':
    app.run(debug=True)