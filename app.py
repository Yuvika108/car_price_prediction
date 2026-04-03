from flask import Flask, render_template, request, redirect
from flask_cors import CORS, cross_origin
import pickle
import pandas as pd
import numpy as np

app=Flask(__name__)
cors=CORS(app)
model=pickle.load(open('LinearRegressionModel.pkl', 'rb'))
df = pd.read_csv('quikr_car.csv')
# Keep only numeric years for the dropdown
df = df[df['year'].astype(str).str.isnumeric()]

@app.route('/', methods=['GET', 'POST'])
def index():
    companies=sorted(df['company'].unique())
    car_models=sorted(df['name'].unique())
    years=sorted(df['year'].unique(), reverse=True)
    fuel_types=sorted(df['fuel_type'].unique())

    companies.insert(0, 'Select Company')
    return render_template('index.html', companies=companies, car_models=car_models, years=years, fuel_types=fuel_types)

@app.route('/get_models', methods=['POST'])
@cross_origin()
def get_models():
    company = request.form.get('company')
    car_models = sorted(df[df['company'] == company]['name'].unique())
    return render_template('model_options.html', car_models=car_models)

@app.route('/predict', methods=['POST'])
@cross_origin()
def predict():

    company=request.form.get('company')

    car_model=request.form.get('car_model')
    year=int(request.form.get('year'))
    fuel_type=request.form.get('fuel_type')
    kms_driven=int(request.form.get('kms_driven'))

    prediction = model.predict(pd.DataFrame(
        [[car_model, company, year, kms_driven, fuel_type]],
        columns=['name', 'company', 'year', 'kms_driven', 'fuel_type']
    ))
    print(prediction)

    # Change this:
    return str(np.round(prediction[0], 2))

if __name__ == "__main__":
    app.run(debug=True, port=5001)