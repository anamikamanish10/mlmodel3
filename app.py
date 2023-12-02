from flask import Flask, render_template, request, jsonify
import numpy as np
from sklearn.linear_model import LinearRegression
import pandas as pd
import joblib

app = Flask(__name__)

# Load data from the modified CSV file
data = pd.read_csv("C:\\Users\\manis\\OneDrive\\Desktop\\data.csv")

X_train = data[['feature1', 'feature2', 'feature3']].values
y_train = data['target'].values

# Train a multiple linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Save the model using joblib
joblib.dump(model, 'multiple_linear_model.joblib')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input data from the form
        feature1 = float(request.form['feature1'])
        feature2 = float(request.form['feature2'])
        feature3 = float(request.form['feature3'])

        # Convert the input data to a NumPy array
        input_data = np.array([feature1, feature2, feature3]).reshape(1, -1)

        # Make a prediction using the multiple linear regression model
        prediction = model.predict(input_data)

        # Return the prediction as JSON
        return jsonify({'prediction': prediction[0]})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run()