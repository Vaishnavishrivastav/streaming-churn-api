from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI()

# Load pipeline
pipeline = joblib.load("churn_pipeline.pkl")

class UserData(BaseModel):
    age: int
    gender: str
    country: str
    plan_type: str
    device_type: str
    minutes_watched: int
    days_since_last_login: int
    favourite_genre: str
    num_movies_watched: int
    num_series_watched: int
    avg_watch_time_per_session: float
    customer_support_calls: int
    payment_status: str
    auto_renew: str

@app.get("/")
def home():
    return {"message": "Streaming Churn Prediction API"}

@app.post("/predict")
def predict(data: UserData):
    df = pd.DataFrame([data.dict()])
    prediction = pipeline.predict(df)[0]
    probability = pipeline.predict_proba(df)[0][1]
    
    return {
        "churn_prediction": "Yes" if prediction == 1 else "No",
        "churn_probability": float(probability)
    }
