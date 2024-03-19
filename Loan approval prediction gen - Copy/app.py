from flask import Flask, render_template, request
import pickle
import pandas as pd

app = Flask(__name__)

# Load the trained decision tree model
# with open('decisionTreeModel.sav', 'rb') as model_file:
#     decision_tree_model = pickle.load(model_file)
    
with open("decisionTreeModel.pkl", "rb") as f:
    decision_tree_model = pickle.load(f)

# Define a function to check if input values are valid
def validate_input(input_data):
    try:
        # Check for blank fields
        for value in input_data.values():
            if not value.strip():  # Check for empty strings
                return False, "Input fields cannot be blank."
        # Check if data types match
        input_data[' no_of_dependents'] = int(input_data[' no_of_dependents'])
        input_data[' income_annum'] = float(input_data[' income_annum'])
        input_data[' loan_amount'] = float(input_data[' loan_amount'])
        input_data[' loan_term'] = int(input_data[' loan_term'])  # Assuming loan term should be integer
        input_data[' cibil_score'] = int(input_data[' cibil_score'])  # Assuming CIBIL score should be integer
        input_data['Movable_assets'] = float(input_data['Movable_assets'])
        input_data['Immovable_assets'] = float(input_data['Immovable_assets'])
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
        ' no_of_dependents': request.form['no_of_dependents'],
        ' education': request.form['education'],
        ' self_employed': request.form['self_employed'],
        ' income_annum': request.form['income_annum'],
        ' loan_amount': request.form['loan_amount'],
        ' loan_term': request.form['loan_term'],
        ' cibil_score': request.form['cibil_score'],
        'Movable_assets': request.form['movable_assets'],
        'Immovable_assets': request.form['immovable_assets']
    }

    # Validate input
    is_valid, message = validate_input(input_data)
    if not is_valid:
        return render_template('error.html', message=message)

    # Make sure feature names match those used during model training
    # You might need to adjust this depending on the actual feature names in your model
    features_used = [' no_of_dependents', ' education', ' self_employed', ' income_annum', 
                     ' loan_amount', ' loan_term', ' cibil_score', 'Movable_assets', 'Immovable_assets']

    # Filter input_data to include only features used during training
    input_data_filtered = {key: input_data[key] for key in features_used}

    print("Filtered input data:", input_data_filtered)  # Print filtered input data for debugging

    # Make the prediction
    prediction = decision_tree_model.predict(pd.DataFrame([input_data_filtered]))[0]
    prediction_result = "Loan has been approved :)" if prediction == 1 else "Loan has been rejected :("

    # Display the result on a new page
    return render_template('result.html', prediction=prediction_result)


# Error handling for 404 - Page Not Found
@app.errorhandler(404)
def page_not_found(error):
    return render_template('404.html'), 404

if __name__ == '__main__':
    app.run(debug=True)
