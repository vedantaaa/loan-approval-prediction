from flask import Flask, render_template, request, redirect, url_for
import pickle
import pandas as pd

app = Flask(__name__)

# Load the trained decision tree model
with open('decisionTreeModel.sav', 'rb') as model_file:
    decision_tree_model = pickle.load(model_file)

# Define a function to check if input values are valid
def validate_input(input_data):
    try:
        # Check for blank fields
        for value in input_data.values():
            if value == "":
                return False, "Input fields cannot be blank."
        # Check if data types match
        input_data['no_of_dependents'] = int(input_data['no_of_dependents'])
        input_data['income_annum'] = float(input_data['income_annum'])
        input_data['loan_amount'] = float(input_data['loan_amount'])
        input_data['loan_term'] = float(input_data['loan_term'])
        input_data['cibil_score'] = float(input_data['cibil_score'])
        input_data['movable_assets'] = float(input_data['movable_assets'])
        input_data['immovable_assets'] = float(input_data['immovable_assets'])
        return True, "Input is valid."
    except ValueError:
        return False, "Invalid input data type."

# Define the route for the home page
@app.route('/')
def home():
    return render_template('index.html')

# Define the route for predicting loan status
@app.route('/predict', methods=['POST'])
def predict():
    # Get input values from the form
    input_data = {
        'no_of_dependents': request.form['no_of_dependents'],
        'education': request.form['education'],
        'self_employed': request.form['self_employed'],
        'income_annum': request.form['income_annum'],
        'loan_amount': request.form['loan_amount'],
        'loan_term': request.form['loan_term'],
        'cibil_score': request.form['cibil_score'],
        'movable_assets': request.form['movable_assets'],
        'immovable_assets': request.form['immovable_assets']
    }

    # Validate input
    is_valid, message = validate_input(input_data)
    if not is_valid:
        return render_template('error.html', message=message)

    # Create a DataFrame with the input values
    input_data['no_of_dependents'] = int(input_data['no_of_dependents'])
    input_data['income_annum'] = float(input_data['income_annum'])
    input_data['loan_amount'] = float(input_data['loan_amount'])
    input_data['loan_term'] = float(input_data['loan_term'])
    input_data['cibil_score'] = float(input_data['cibil_score'])
    input_data['movable_assets'] = float(input_data['movable_assets'])
    input_data['immovable_assets'] = float(input_data['immovable_assets'])

    # Make the prediction
    prediction = decision_tree_model.predict(pd.DataFrame([input_data]))[0]

    # Display the result on a new page
    return render_template('result.html', prediction=prediction)

# Error handling for 404 - Page Not Found
@app.errorhandler(404)
def page_not_found(error):
    return render_template('404.html'), 404

if __name__ == '__main__':
    app.run(debug=True)
