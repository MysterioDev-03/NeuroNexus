from flask import Flask, render_template, request
import joblib
import numpy as np

# Import the model class so joblib can unpickle it
from model_training import LogisticRegressionModel

app = Flask(__name__, static_folder='static')

# Load model and scaling values
try:
    model, mean, std = joblib.load("titanic_model.pkl")
except Exception as e:
    print("Error loading the model:", e)
    model, mean, std = None, None, None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Validate form inputs and handle missing or invalid values
        pclass = int(request.form['pclass']) if request.form['pclass'].isdigit() else None
        gender = 1 if request.form['gender'] == 'female' else 0
        age = float(request.form['age']) if request.form['age'].replace('.', '', 1).isdigit() else None
        sibsp = int(request.form['sibsp']) if request.form['sibsp'].isdigit() else None
        parch = int(request.form['parch']) if request.form['parch'].isdigit() else None
        fare = float(request.form['fare']) if request.form['fare'].replace('.', '', 1).isdigit() else None
        embarked = {'C': 0, 'Q': 1, 'S': 2}.get(request.form['embarked'], None)

        # Ensure all fields are filled correctly
        if None in [pclass, age, sibsp, parch, fare, embarked]:
            return render_template('index.html', prediction="Please fill all fields correctly.")

        # Prepare the input data
        input_data = np.array([[pclass, gender, age, sibsp, parch, fare, embarked]])
        input_data = (input_data - mean) / std  # Standardize based on mean and std

        # Model prediction
        prediction = model.predict(input_data)
        result = "Survived" if prediction[0] == 1 else "Did not survive"
        return render_template('index.html', prediction=result)
    except Exception as e:
        print("Error during prediction:", e)
        return render_template('index.html', prediction="An error occurred. Please try again.")

if __name__ == '__main__':
    app.run(debug=True)
