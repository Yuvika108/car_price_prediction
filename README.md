# car_price_prediction
Car Price Prediction
Project Overview

This project predicts the selling price of used cars based on features such as brand, year, mileage, fuel type, and transmission. It uses machine learning to provide accurate price estimations.

Dataset

The dataset quikr_car.csv contains information about used cars:

Brand – Car manufacturer
Year – Year of manufacture
Mileage – Distance traveled by the car
Fuel Type – Petrol, Diesel, CNG, etc.
Transmission – Manual or Automatic
Price – Selling price (target variable)
Other Features – Engine size, seats, etc.
Features
Data preprocessing (handling missing values, encoding categorical variables)
Exploratory data analysis (EDA)
Machine learning model (Linear Regression)
Interactive Flask web app for predictions
Installation

Clone the repository:

git clone <repository_url>
cd car_price_prediction

Install required packages:

pip install -r requirements.txt

Run the Flask app:

python app.py
Open your browser at http://127.0.0.1:5000/
Usage
Enter car details (year, brand, mileage, fuel type, transmission) in the web app
Click Predict
Get the estimated selling price
Model
Algorithm: Linear Regression
Libraries: scikit-learn, pandas, numpy
Model File: LinearRegressionModel.pkl
Folder Structure
car_price_prediction/
│
├── app.py
├── LinearRegressionModel.pkl
├── quikr_car.csv
├── templates/
│   └── index.html
├── static/
│   └── style.css
├── README.md
└── requirements.txt
Dependencies
Python 3.x
Flask
Pandas
Numpy
Scikit-learn
Flask-CORS
Author

Srishti Mishra

License

MIT License
