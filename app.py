
from flask import Flask, render_template, request
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

app = Flask(__name__)

# Load dataset
data = pd.read_csv('dataset/blood.csv')
df = pd.DataFrame(data)

# Features and target
X = df[['Last_Donation', 'Total_Donations',
        'Total_Blood_Donated', 'Body_Weight']]

Y = df['Prediction']

# Split dataset
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, train_size=0.8, random_state=2
)

# Train model
model = DecisionTreeClassifier(
    criterion='entropy', random_state=2
)
model.fit(X_train, Y_train)

# Calculate accuracy
y_test_pred = model.predict(X_test)
accuracy = accuracy_score(Y_test, y_test_pred) * 100


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():

    last_donation = float(request.form['last_donation'])
    total_donations = int(request.form['total_donations'])
    total_blood_donated = float(request.form['total_blood_donated'])
    body_weight = float(request.form['body_weight'])

    prediction = model.predict([[
        last_donation,
        total_donations,
        total_blood_donated,
        body_weight
    ]])

    if prediction[0] == 1:
        result = "LIKELY TO DONATE"
    else:
        result = "UNLIKELY TO DONATE"

    return render_template(
        'result.html',
        prediction=result,
        accuracy=round(accuracy, 2)
    )


if __name__ == '__main__':
    app.run(debug=True)
