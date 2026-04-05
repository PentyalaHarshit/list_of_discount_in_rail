from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import pandas as pd
import numpy as np
from datetime import datetime
import joblib
import os
import logging
import sqlite3

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DB_FILE = "tickets.db"


# =========================================================
# LOAD MODELS
# OLD DATE-ONLY MODELS
# =========================================================
def load_models():
    return {
        "xgb_item": joblib.load("xgb_item.pkl"),
        "xgb_train": joblib.load("xgb_train.pkl"),
        "xgb_reg_discount": joblib.load("xgb_reg_discount.pkl"),
        "xgb_reg_price": joblib.load("xgb_reg_price.pkl"),
        "mlb_item": joblib.load("mlb_item.pkl"),
        "mlb_train": joblib.load("mlb_train.pkl"),
    }


# =========================================================
# DB SETUP
# =========================================================
def init_db():
    conn = sqlite3.connect(DB_FILE)
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS ticket_prices (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            departure TEXT NOT NULL,
            arrival TEXT NOT NULL,
            base_price REAL NOT NULL
        )
    """)

    cur.execute("SELECT COUNT(*) FROM ticket_prices")
    count = cur.fetchone()[0]

    if count == 0:
        sample_prices = [
            ("Lyon", "Paris", 120.0),
            ("Paris", "Lyon", 115.0),
            ("Lyon", "Marseille", 95.0),
            ("Paris", "Marseille", 140.0),
            ("Berlin", "Paris", 180.0),
            ("Brussels", "Paris", 110.0),
            ("Amsterdam", "Paris", 150.0),
            ("DB", "TGV", 90.0),
            ("Eurostar", "Thalys", 130.0),
        ]
        cur.executemany("""
            INSERT INTO ticket_prices (departure, arrival, base_price)
            VALUES (?, ?, ?)
        """, sample_prices)

    conn.commit()
    conn.close()


def get_base_price(departure: str, arrival: str):
    conn = sqlite3.connect(DB_FILE)
    cur = conn.cursor()
    cur.execute("""
        SELECT base_price
        FROM ticket_prices
        WHERE LOWER(departure) = LOWER(?)
          AND LOWER(arrival) = LOWER(?)
        LIMIT 1
    """, (departure, arrival))
    row = cur.fetchone()
    conn.close()
    return row[0] if row else None


# =========================================================
# DATE FEATURES FOR DISCOUNT PREDICTION
# =========================================================
def create_features_from_date(date_str: str) -> pd.DataFrame:
    date_obj = datetime.strptime(date_str, "%Y-%m-%d")
    return pd.DataFrame({
        "Year": [date_obj.year],
        "Month": [date_obj.month],
        "Day": [date_obj.day],
        "Weekday": [date_obj.weekday()],
        "IsWeekend": [1 if date_obj.weekday() >= 5 else 0],
        "Season": [((date_obj.month % 12) + 3) // 3 - 1],
        "Month_sin": [np.sin(2 * np.pi * date_obj.month / 12)],
        "Month_cos": [np.cos(2 * np.pi * date_obj.month / 12)]
    })


# =========================================================
# APP
# =========================================================
def create_app(load_real_models: bool = True) -> FastAPI:
    app = FastAPI()

    init_db()

    if os.path.exists("static"):
        app.mount("/static", StaticFiles(directory="static"), name="static")

    if load_real_models:
        app.state.models = load_models()
    else:
        app.state.models = None

    # -----------------------------------------------------
    # PAGE 1
    # -----------------------------------------------------
    @app.get("/", response_class=HTMLResponse)
    def home():
        return """
        <html>
        <head>
            <title>Rail Platform</title>
            <style>
                body {
                    font-family: Arial, sans-serif;
                    background: #f4f7fb;
                    margin: 0;
                    padding: 0;
                }
                .container {
                    width: 520px;
                    margin: 60px auto;
                    background: white;
                    padding: 30px;
                    border-radius: 12px;
                    box-shadow: 0 0 15px rgba(0,0,0,0.12);
                }
                h2 {
                    text-align: center;
                    margin-bottom: 25px;
                    color: #1f3c88;
                }
                label {
                    display: block;
                    margin-top: 12px;
                    margin-bottom: 6px;
                    font-weight: bold;
                }
                input {
                    width: 100%;
                    padding: 10px;
                    border: 1px solid #ccc;
                    border-radius: 6px;
                    margin-bottom: 10px;
                    box-sizing: border-box;
                }
                .btn-row {
                    display: flex;
                    gap: 10px;
                    margin-top: 18px;
                }
                button {
                    flex: 1;
                    padding: 12px;
                    border: none;
                    border-radius: 6px;
                    color: white;
                    font-size: 15px;
                    font-weight: bold;
                    cursor: pointer;
                }
                .without-btn {
                    background: #007bff;
                }
                .discount-btn {
                    background: #28a745;
                }
                button:hover {
                    opacity: 0.92;
                }
            </style>
        </head>
        <body>
            <div class="container">
                <h2>Rail Platform</h2>

                <form action="/choose-flow" method="post">
                    <label for="departure">Departure</label>
                    <input type="text" id="departure" name="departure" required>

                    <label for="arrival">Arrival</label>
                    <input type="text" id="arrival" name="arrival" required>

                    <label for="travel_date">Travel Date</label>
                    <input type="date" id="travel_date" name="travel_date" required>

                    <div class="btn-row">
                        <button type="submit" name="action" value="without_discount" class="without-btn">
                            Without Discount
                        </button>
                        <button type="submit" name="action" value="discount" class="discount-btn">
                            Discount
                        </button>
                    </div>
                </form>
            </div>
        </body>
        </html>
        """

    # -----------------------------------------------------
    # CHOOSE FLOW
    # -----------------------------------------------------
    @app.post("/choose-flow", response_class=HTMLResponse)
    def choose_flow(
        departure: str = Form(...),
        arrival: str = Form(...),
        travel_date: str = Form(...),
        action: str = Form(...)
    ):
        if action == "without_discount":
            return render_ticket_price_page(departure, arrival, travel_date)
        return render_discount_details_page(departure, arrival, travel_date)

    # -----------------------------------------------------
    # WITHOUT DISCOUNT -> TICKET PRICE PAGE
    # -----------------------------------------------------
    def render_ticket_price_page(departure: str, arrival: str, travel_date: str):
        base_price = get_base_price(departure, arrival)

        if base_price is None:
            return f"""
            <html>
            <body style="font-family: Arial; padding: 30px;">
                <h2>Route Not Found</h2>
                <p>No price found for <b>{departure}</b> to <b>{arrival}</b>.</p>
                <a href="/">Go Back</a>
            </body>
            </html>
            """

        return f"""
        <html>
        <head>
            <title>Ticket Price</title>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    background: #f4f7fb;
                    margin: 0;
                    padding: 0;
                }}
                .container {{
                    width: 650px;
                    margin: 60px auto;
                    background: white;
                    padding: 30px;
                    border-radius: 12px;
                    box-shadow: 0 0 15px rgba(0,0,0,0.12);
                }}
                table {{
                    width: 100%;
                    border-collapse: collapse;
                    margin-top: 20px;
                }}
                th, td {{
                    border: 1px solid #ddd;
                    padding: 12px;
                    text-align: left;
                }}
                th {{
                    background: #1f3c88;
                    color: white;
                }}
                a {{
                    display: inline-block;
                    margin-top: 20px;
                    text-decoration: none;
                    color: #333;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <h2>Ticket Price</h2>
                <p><b>Departure:</b> {departure}</p>
                <p><b>Arrival:</b> {arrival}</p>
                <p><b>Travel Date:</b> {travel_date}</p>

                <table>
                    <tr>
                        <th>Route</th>
                        <th>Base Price</th>
                    </tr>
                    <tr>
                        <td>{departure} → {arrival}</td>
                        <td>€{base_price:.2f}</td>
                    </tr>
                </table>

                <a href="/">Go Back</a>
            </div>
        </body>
        </html>
        """

    # -----------------------------------------------------
    # DISCOUNT -> PAGE 2
    # NAME, PHONE, DISCOUNT DATE
    # -----------------------------------------------------
    def render_discount_details_page(departure: str, arrival: str, travel_date: str):
        return f"""
        <html>
        <head>
            <title>Discount Details</title>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    background: #f4f7fb;
                    margin: 0;
                    padding: 0;
                }}
                .container {{
                    width: 520px;
                    margin: 60px auto;
                    background: white;
                    padding: 30px;
                    border-radius: 12px;
                    box-shadow: 0 0 15px rgba(0,0,0,0.12);
                }}
                h2 {{
                    text-align: center;
                    margin-bottom: 25px;
                    color: #1f3c88;
                }}
                label {{
                    display: block;
                    margin-top: 12px;
                    margin-bottom: 6px;
                    font-weight: bold;
                }}
                input {{
                    width: 100%;
                    padding: 10px;
                    border: 1px solid #ccc;
                    border-radius: 6px;
                    margin-bottom: 10px;
                    box-sizing: border-box;
                }}
                button {{
                    width: 100%;
                    padding: 12px;
                    border: none;
                    border-radius: 6px;
                    color: white;
                    font-size: 15px;
                    font-weight: bold;
                    cursor: pointer;
                    background: #28a745;
                    margin-top: 18px;
                }}
                a {{
                    display: inline-block;
                    margin-top: 15px;
                    text-decoration: none;
                    color: #333;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <h2>Discount Details</h2>

                <p><b>Departure:</b> {departure}</p>
                <p><b>Arrival:</b> {arrival}</p>
                <p><b>Travel Date:</b> {travel_date}</p>

                <form action="/predict-discount" method="post">
                    <input type="hidden" name="departure" value="{departure}">
                    <input type="hidden" name="arrival" value="{arrival}">
                    <input type="hidden" name="travel_date" value="{travel_date}">

                    <label for="name">Name</label>
                    <input type="text" id="name" name="name" required>

                    <label for="phone">Phone</label>
                    <input type="tel" id="phone" name="phone" required>

                    <label for="discount_date">Discount Date</label>
                    <input type="date" id="discount_date" name="discount_date" required>

                    <button type="submit">Predict Discount</button>
                </form>

                <a href="/">Go Back</a>
            </div>
        </body>
        </html>
        """

    # -----------------------------------------------------
    # DISCOUNT PREDICTION RESULT
    # PREDICT ONLY USING DISCOUNT DATE
    # -----------------------------------------------------
    @app.post("/predict-discount", response_class=HTMLResponse)
    def predict_discount(
        departure: str = Form(...),
        arrival: str = Form(...),
        travel_date: str = Form(...),
        name: str = Form(...),
        phone: str = Form(...),
        discount_date: str = Form(...)
    ):
        logger.info(f"Discount prediction requested for discount date: {discount_date}")

        try:
            X_new = create_features_from_date(discount_date)

            models = app.state.models
            xgb_item = models["xgb_item"]
            xgb_train = models["xgb_train"]
            xgb_reg_discount = models["xgb_reg_discount"]
            xgb_reg_price = models["xgb_reg_price"]
            mlb_item = models["mlb_item"]
            mlb_train = models["mlb_train"]

            pred_items = mlb_item.inverse_transform(xgb_item.predict(X_new))
            pred_trains = mlb_train.inverse_transform(xgb_train.predict(X_new))
            pred_discounts = xgb_reg_discount.predict(X_new)
            pred_prices = xgb_reg_price.predict(X_new)

            html_output = f"""
            <html>
            <head>
                <title>Discount Prediction</title>
                <style>
                    body {{
                        font-family: Arial, sans-serif;
                        background: #f4f7fb;
                        margin: 0;
                        padding: 0;
                    }}
                    .container {{
                        width: 760px;
                        margin: 50px auto;
                        background: white;
                        padding: 30px;
                        border-radius: 12px;
                        box-shadow: 0 0 15px rgba(0,0,0,0.12);
                    }}
                    h2 {{
                        color: #1f3c88;
                        margin-bottom: 20px;
                    }}
                    .card {{
                        background: #eaf7ed;
                        border-left: 6px solid #28a745;
                        padding: 15px;
                        border-radius: 8px;
                        margin-bottom: 12px;
                        font-size: 16px;
                    }}
                    a {{
                        display: inline-block;
                        margin-top: 15px;
                        text-decoration: none;
                        color: #333;
                    }}
                </style>
            </head>
            <body>
                <div class="container">
                    <h2>Predicted Discount List</h2>

                    <p><b>Name:</b> {name}</p>
                    <p><b>Phone:</b> {phone}</p>
                    <p><b>Departure:</b> {departure}</p>
                    <p><b>Arrival:</b> {arrival}</p>
                    <p><b>Travel Date:</b> {travel_date}</p>
                    <p><b>Discount Date:</b> {discount_date}</p>
            """

            if len(pred_items[0]) == 0 or len(pred_trains[0]) == 0:
                html_output += """
                    <h3>No discounts predicted for this discount date.</h3>
                    <a href="/">Go Back</a>
                </div>
            </body>
            </html>
                """
                return html_output

            discount_list = [
                f"{item} at {train}: {discount:.0f}% off, now €{price:.2f}"
                for item, train, discount, price in zip(
                    pred_items[0],
                    pred_trains[0],
                    pred_discounts[0],
                    pred_prices[0]
                )
            ]

            for d in discount_list:
                html_output += f'<div class="card">{d}</div>'

            html_output += """
                    <a href="/">Go Back</a>
                </div>
            </body>
            </html>
            """
            return html_output

        except Exception as e:
            return f"""
            <html>
            <body style="font-family: Arial; padding: 30px;">
                <h2>Error</h2>
                <p>{str(e)}</p>
                <a href="/">Go Back</a>
            </body>
            </html>
            """

    return app


app = create_app(load_real_models=True)