from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__, template_folder='.')  # Use current directory for templates

model = pickle.load(open('model.pkl', 'rb'))

# List of features expected by the model
FEATURES = [
    'Age', 'Occupation', 'Assets', 'Investments', 'Savings', 'Debt',
    'Gross monthly income', 'Net monthly income', 'Rent/Mortgage', 'Utilities',
    'Emergency Fund', 'Dining out', 'Groceries', 'Miscellaneous', '% Expenses', '% Savings'
]

@app.route('/')
def home():
    return render_template('front.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    monthly_expenses = float(data.get('monthly_expenses', 0))
    current_insurance_coverage = float(data.get('current_insurance_coverage', 0))

    # Map frontend fields to model features
    # You may need to adjust this mapping based on your frontend form
    input_dict = {
        'Age': int(data.get('age', 0)),
        'Occupation': data.get('occupation', ''),
        'Assets': float(data.get('assets', 0)),
        'Investments': float(data.get('investments', 0)),
        'Savings': float(data.get('savings', 0)),
        'Debt': float(data.get('debt', 0)),
        'Gross monthly income': float(data.get('income', 0)),
        'Net monthly income': float(data.get('income', 0)),  # Use same as gross if not separate
        'Rent/Mortgage': monthly_expenses,
        'Utilities': monthly_expenses,
        'Emergency Fund': float(data.get('emergency_fund', 0)),
        'Dining out': monthly_expenses,
        'Groceries': monthly_expenses,
        'Miscellaneous': monthly_expenses,
        '% Expenses': float(data.get('percent_expenses', 0)),
        '% Savings': float(data.get('percent_savings', 0))
    }

    # Create DataFrame for model input
    input_df = pd.DataFrame([input_dict], columns=FEATURES)

    # Predict total insurance need using the model
    predicted_insurance_need = model.predict(input_df)[0]
    
    # Calculate insurance gap
    insurance_gap = predicted_insurance_need - current_insurance_coverage

    # Return the prediction and insurance gap in the response
    return jsonify({
        'prediction': float(predicted_insurance_need),
        'insurance_gap': float(insurance_gap)
    })

if __name__ == '__main__':
    app.run(debug=True)
