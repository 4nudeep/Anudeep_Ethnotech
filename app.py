
import os
from pathlib import Path

import pandas as pd
from flask import Flask, render_template, request
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

app = Flask(__name__)

# Dataset path
BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "dataset" / "blood.csv"

# Load dataset
if not DATASET_PATH.exists():
    raise FileNotFoundError(
        f"Dataset not found: {DATASET_PATH}"
    )

df = pd.read_csv(DATASET_PATH)

# Features and target
features = [
    'Last_Donation',
    'Total_Donations',
    'Total_Blood_Donated',
    'Body_Weight'
]

X = df[features]
Y = df['Prediction']

# Split dataset
X_train, X_test, Y_train, Y_test = train_test_split(
    X,
    Y,
    train_size=0.8,
    random_state=2
)

# Train Decision Tree model
model = DecisionTreeClassifier(
    criterion='entropy',
    random_state=2
)

model.fit(X_train, Y_train)

# Calculate accuracy
y_test_pred = model.predict(X_test)
accuracy = accuracy_score(Y_test, y_test) * 100


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        last_donation = float(request.form['last_donation'])
        total_donations = int(request.form['total_donations'])
        total_blood_donated = float(
            request.form['total_blood_donated']
        )
        body_weight = float(request.form['body_weight'])

        if min(
            last_donation,
            total_donations,
            total_blood_donated,
            body_weight
        ) < 0:
            raise ValueError("Values cannot be negative.")

        # Predict using DataFrame
        input_data = pd.DataFrame(
            [[
                last_donation,
                total_donations,
                total_blood_donated,
                body_weight
            ]],
            columns=features
        )

        prediction = model.predict(input_data)[0]

        if prediction == 1:
            result = "LIKELY TO DONATE"
        else:
            result = "UNLIKELY TO DONATE"

        return render_template(
            'result.html',
            prediction=result,
            accuracy=round(accuracy, 2)
        )

    except (ValueError, TypeError, KeyError):
        return render_template(
            'index.html',
            error="Please enter valid, non-negative values."
        ), 400


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
