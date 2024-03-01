import pickle
from flask import Flask, render_template, request
from flask_wtf.csrf import CSRFProtect
import numpy as np
import pandas as pd

app = Flask(__name__)  # Creating Flask application instance
csrf = CSRFProtect(app)  # Adding CSRF protection to the app

# Loading the trained model
t_model = pickle.load(open("models/model.pkl", "rb"))  # Loading  model

@app.route('/')  # Defining route for the home page
def home():
    """
    Renders the home page.

    Returns:
        rendered HTML template
    """
    return render_template('index.html')  # Rendering the HTML template for the home page

@app.route('/detect', methods=['POST'])  # Defining route for phishing detection
@csrf.exempt  # Exempting CSRF protection for this route

if __name__ == '__main__':
    app.run(debug=True)  # Running the Flask application in debug mode
