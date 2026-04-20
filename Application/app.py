import os
from flask import Flask, request, render_template
import numpy as np
import pickle

# Paths
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_DIR = os.path.join(BASE_DIR, "Model")

app = Flask(__name__)

# Load model artifacts
model_path = os.path.join(MODEL_DIR, "lending_club_pipeline.pkl")

if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found at {model_path}")

with open(model_path, "rb") as file:
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

#Calculate the EMI
def calculate_emi(P, r, n):
    r = r / (12 * 100)  # monthly interest
    emi = (P * r * (1 + r)**n) / ((1 + r)**n - 1)
    return emi

#Best Loan term to be predicted
def optimize_loan_term(amount, risk_score, dti, state, emp_len):
    best_term = None
    best_score = float('inf')

    for term in [12, 24, 36, 60]:
        emi = calculate_emi(amount, 10, term)  # assume 10% interest
        
        # Risk penalty (example logic)
        risk_penalty = dti * 10 + (700 - risk_score)

        # Higher term = more risk
        term_penalty = term * 0.5

        score = emi + risk_penalty + term_penalty

        if score < best_score:
            best_score = score
            best_term = term

    return best_term

# Home page
@app.route('/')
def index():
    return render_template('index.html')

term_map = {
    0: 60,
    1: 36,
    2: 24,
    3: 12
}
# Prediction
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input
        req_amount = float(request.form.get('amount', 0))
        fico = float(request.form.get('risk_score', 0))
        dti = float(request.form.get('dti', 0))
        state_raw = request.form.get('state', 'NY')
        # Load kmeans
        kmeans = artifacts['kmeans']
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
        #best_term = optimize_loan_term(final_limit, fico, dti, state_enc, emp_len)//Take from K-Means
        segment = kmeans.predict(scaler.transform([[req_amount, fico, dti, state_enc, emp_len]]))[0]
        best_term = term_map.get(int(segment), 36)
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
            loan_term=best_term,
            color=color
        )

    except Exception as e:
        return f"Error: {str(e)}"


# Run app
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)