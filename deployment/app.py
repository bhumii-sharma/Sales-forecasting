from flask import Flask, render_template, request, flash, redirect, url_for, session
from werkzeug.utils import secure_filename
import os
import pandas as pd
from deployment.pipeline.prediction_pipeline import PredictionPipeline  # Assuming this handles sales forecasting logic
import json

app = Flask(__name__)
app.secret_key = os.urandom(24)  # Add secret key for flashing

# Allowed file extensions for uploaded CSV
ALLOWED_EXTENSIONS = {'csv'}

# Function to check if the uploaded file is an allowed CSV format
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["GET", "POST"])
def prediction():
    if request.method == "POST":
        file = request.files['file']
        
        if file and allowed_file(file.filename):
            # Save the file temporarily
            filename = secure_filename(file.filename)
            file_path = os.path.join("uploads", filename)
            file.save(file_path)
            
            # Read the CSV file
            data = pd.read_csv(file_path)
            
            # Now pass the data through the prediction pipeline
            prediction_pipeline = PredictionPipeline()
            predicted_output = prediction_pipeline.forecast_sales(data)  # Assuming this method forecasts sales

            # Render the prediction result page with the forecasted sales or price
            return render_template("predict.html", forecasted_output=predicted_output)
        else:
            flash('Invalid file format. Please upload a .csv file.')
            return redirect(request.url)

    return render_template("upload_csv.html")

if __name__ == "__main__":
    app.run(port=5000)
