from flask import Flask, render_template, request
import pickle
import pandas as pd

app = Flask(__name__)

# Load the trained decision tree model
with open('decisionTreeModel.sav', 'rb') as model_file:
    decision_tree_model = pickle.load(model_file)

# Define the route for the home page
@app.route('/')
def home():
    return render_template('index.html')

# Define the route for predicting loan status
@app.route('/predict', methods=['POST'])
def predict():
    # Get input values from the form
    no_of_dependents = int(request.form['no_of_dependents'])
    education = str(request.form['education'])
    self_employed = str(request.form['self_employed'])
    income_annum = float(request.form['income_annum'])
    loan_amount = float(request.form['loan_amount'])
    loan_term = float(request.form['loan_term'])
    cibil_score = float(request.form['cibil_score'])
    movable_assets = float(request.form['movable_assets'])
    immovable_assets = float(request.form['immovable_assets'])

    # Create a DataFrame with the input values
    input_data = pd.DataFrame({
        ' no_of_dependents': [no_of_dependents],
        ' education': [education],
        ' self_employed': [self_employed],
        ' income_annum': [income_annum],
        ' loan_amount': [loan_amount],
        ' loan_term': [loan_term],
        ' cibil_score': [cibil_score],
        'Movable_assets': [movable_assets],
        'Immovable_assets': [immovable_assets]
    })

    # Make the prediction
    prediction = decision_tree_model.predict(input_data)[0]

    # Display the result on a new page
    return render_template('result.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
