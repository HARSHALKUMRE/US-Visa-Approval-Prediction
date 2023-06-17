from flask import Flask, request, render_template, jsonify
from flask_cors import CORS, cross_origin
from visa.pipeline.training_pipeline import TrainingPipeline
from visa.pipeline.prediction_pipeline import USVisaData, USVisaPredictor 
from visa.exception import CustomException
from visa.constant import CONFIG_DIR, get_current_time_stamp
import os
import sys

ROOT_DIR = os.getcwd()
SAVED_MODELS_DIR_NAME = "saved_models/"
MODEL_DIR = os.path.join(ROOT_DIR, SAVED_MODELS_DIR_NAME)

application = Flask(__name__)
app = application

@app.route('/')
@cross_origin()
def home_page():
    return render_template('index.html')

@app.route('/predict', methods=['GET','POST'])
@cross_origin()
def predict_datapoint():
    if request.method == 'GET':
        return render_template('index.html')

    else:
        data = USVisaData(
            continent = request.form.get('continent'),
            education_of_employee = request.form.get('education_of_employee'),
            has_job_experience = request.form.get('has_job_experience'),
            requires_job_training = request.form.get('requires_job_training'),
            no_of_employees = int(request.form.get('no_of_employees')),
            company_age = int(request.form.get('company_age')),
            region_of_employment = request.form.get('region_of_employment'),
            prevailing_wage = float(request.form.get('prevailing_wage')),
            unit_of_wage = request.form.get('unit_of_wage'),
            full_time_position = request.form.get('full_time_position')

        )

        pred_df = data.get_us_visa_input_data_frame()

        print(pred_df)

        predict_pipeline = USVisaPredictor(model_dir=MODEL_DIR)
        pred = predict_pipeline.predict(X=pred_df)[0]
        if pred == 1:
            results = "The US Visa Approved"
        else:
            results = "The US Visa not Approved"
        return render_template('index.html', results=results, pred_df=pred_df)

@app.route('/train')
@cross_origin()
def trainRoute():
    try:
        training_pipeline = TrainingPipeline()

        training_pipeline.run_pipeline()

        return "Training is successful!"
    except Exception as e:
        raise CustomException(e, sys)




if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)