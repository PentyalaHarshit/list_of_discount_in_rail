import sys
import os
import sqlite3
import tempfile
import numpy as np
from fastapi.testclient import TestClient

# adjust import path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import app as app_module


# =========================================================
# DUMMY MODELS
# =========================================================
class DummyMLB:
    def inverse_transform(self, arr):
        return [("Laptop", "Shoes")]


class DummyClassifier:
    def predict(self, X):
        return np.array([[1, 1]])


class DummyRegressor:
    def predict(self, X):
        return np.array([[20.0, 35.0]])


# =========================================================
# TEST DB SETUP
# =========================================================
def create_test_db():
    temp_db = tempfile.NamedTemporaryFile(delete=False, suffix=".db")
    temp_db.close()

    conn = sqlite3.connect(temp_db.name)
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE ticket_prices (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            departure TEXT NOT NULL,
            arrival TEXT NOT NULL,
            base_price REAL NOT NULL
        )
    """)

    cur.execute("""
        INSERT INTO ticket_prices (departure, arrival, base_price)
        VALUES (?, ?, ?)
    """, ("Lyon", "Paris", 120.0))

    conn.commit()
    conn.close()

    return temp_db.name


# =========================================================
# TEST APP FACTORY
# =========================================================
def get_test_client():
    test_db = create_test_db()
    app_module.DB_FILE = test_db

    app = app_module.create_app(load_real_models=False)

    app.state.models = {
        "xgb_item": DummyClassifier(),
        "xgb_train": DummyClassifier(),
        "xgb_reg_discount": DummyRegressor(),
        "xgb_reg_price": DummyRegressor(),
        "mlb_item": DummyMLB(),
        "mlb_train": DummyMLB(),
    }

    client = TestClient(app)
    return client, test_db


# =========================================================
# TEST HOME PAGE
# =========================================================
def test_home_page():
    client, test_db = get_test_client()

    response = client.get("/")

    assert response.status_code == 200
    assert "Rail Platform" in response.text
    assert "<form" in response.text
    assert "Departure" in response.text
    assert "Arrival" in response.text
    assert "Travel Date" in response.text

    os.remove(test_db)


# =========================================================
# TEST WITHOUT DISCOUNT FLOW
# =========================================================
def test_without_discount_flow():
    client, test_db = get_test_client()

    response = client.post(
        "/choose-flow",
        data={
            "departure": "Lyon",
            "arrival": "Paris",
            "travel_date": "2026-04-10",
            "action": "without_discount"
        }
    )

    assert response.status_code == 200
    assert "Ticket Price" in response.text
    assert "Lyon" in response.text
    assert "Paris" in response.text
    assert "€120.00" in response.text

    os.remove(test_db)


# =========================================================
# TEST DISCOUNT PAGE 2
# =========================================================
def test_discount_details_page():
    client, test_db = get_test_client()

    response = client.post(
        "/choose-flow",
        data={
            "departure": "Lyon",
            "arrival": "Paris",
            "travel_date": "2026-04-10",
            "action": "discount"
        }
    )

    assert response.status_code == 200
    assert "Discount Details" in response.text
    assert "Lyon" in response.text
    assert "Paris" in response.text
    assert "Discount Date" in response.text
    assert "Name" in response.text
    assert "Phone" in response.text

    os.remove(test_db)


# =========================================================
# TEST DISCOUNT PREDICTION RESULT
# =========================================================
def test_predict_discount():
    client, test_db = get_test_client()

    response = client.post(
        "/predict-discount",
        data={
            "departure": "Lyon",
            "arrival": "Paris",
            "travel_date": "2026-04-10",
            "name": "Harshit",
            "phone": "9876543210",
            "discount_date": "2026-04-04"
        }
    )

    assert response.status_code == 200
    assert "Predicted Discount List" in response.text
    assert "Harshit" in response.text
    assert "9876543210" in response.text
    assert "Laptop" in response.text
    assert "Shoes" in response.text

    os.remove(test_db)