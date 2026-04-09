Train Discount Prediction System

Description:
This project is a FastAPI-based web application that predicts train discounts and ticket prices using machine learning models based on route and travel date.

--------------------------------------------------

Features:
- User input (Name, Phone)
- Select source and destination
- Enter travel date
- Predict:
  - Train name
  - Discount percentage
  - Final ticket price
- Displays results as discount cards
- Route-based train filtering

--------------------------------------------------

Project Structure:

app.py                  -> Main FastAPI application
routes.py               -> Route handling
requirements.txt        -> Dependencies
Dockerfile              -> Docker setup

static/                 -> Images and static files
models/                 -> ML model files (.pkl)
data/                   -> Dataset files (.csv)
test/                   -> Test files
tickets.db              -> Database file

--------------------------------------------------

How It Works:

1. User enters name and phone number
2. Selects source and destination
3. Enters travel date
4. ML models predict:
   - Train
   - Discount %
   - Ticket price
5. Results are displayed to the user

--------------------------------------------------

Example:

Input:
Source: Paris
Destination: Lyon
Date: 2026-04-10

Output:
Train: TGV   | Discount: 25% | Price: 150
Train: SNCF  | Discount: 20% | Price: 170

--------------------------------------------------

Technologies Used:

- Python
- FastAPI
- Machine Learning (XGBoost)
- Joblib
- HTML / CSS / JavaScript
- Docker

--------------------------------------------------

Installation:

1. Clone repository:
   git clone https://github.com/your-username/train-discount-prediction.git
   cd train-discount-prediction

2. Create virtual environment:
   python -m venv venv

3. Activate environment:
   Windows: venv\Scripts\activate
   Mac/Linux: source venv/bin/activate

4. Install dependencies:
   pip install -r requirements.txt

--------------------------------------------------

Run Application:

uvicorn app:app --reload

Open browser:
http://127.0.0.1:8000

--------------------------------------------------

Docker Run:

docker build -t train-discount-app .
docker run -p 8000:8000 train-discount-app

--------------------------------------------------

ML Models:

train_model.pkl          -> Predict train
discount_model.pkl       -> Predict discount
price_model.pkl          -> Predict price
feature_columns.pkl      -> Feature structure

--------------------------------------------------

Future Improvements:

- Add database storage
- Improve UI design
- Add more routes and trains
- Deploy to cloud

--------------------------------------------------

Author:
Harshit Pentyala

--------------------------------------------------

Note:
This project is for learning and demonstration purposes.