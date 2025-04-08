from flask import Flask, request, render_template
import pickle
import numpy as np

# Load the trained AdaBoost model
with open("aboost_clf.pkl", "rb") as f:
    model = pickle.load(f)

# Example categories for 'purpose' — update as per your model training
purpose_categories = ['credit_card', 'debt_consolidation', 'educational', 'home_improvement',
                      'major_purchase', 'small_business', 'all_other']

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Extract numerical features
        numeric_keys = [
            'credit.policy', 'int.rate', 'installment', 'log.annual.inc',
            'dti', 'fico', 'days.with.cr.line', 'revol.bal', 'revol.util',
            'inq.last.6mths', 'delinq.2yrs', 'pub.rec'
        ]
        numerical_features = [float(request.form[key]) for key in numeric_keys]

        # One-hot encode 'purpose'
        purpose_input = request.form['purpose']
        purpose_encoded = [1 if purpose_input == category else 0 for category in purpose_categories]

        # Combine all features
        features = np.array(numerical_features + purpose_encoded).reshape(1, -1)

        # Make prediction
        prediction = model.predict(features)[0]

        return render_template("index.html", prediction=f"Predicted Class: {prediction}")
    
    except Exception as e:
        return render_template("index.html", prediction=f"Error: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)
