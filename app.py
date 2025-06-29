from flask import Flask, request, render_template
import pandas as pd
from src.data_loader import load_data
from src.preprocess import preprocess_data
from src.predict import predict_from_input
from src.model import train_model
import os
import joblib

app = Flask(__name__)

# ✅ Setup everything before app starts
df = load_data("data/Thyroid_new_data.csv")
X, y, encoder = preprocess_data(df)

model_path = "saved_model/xgboost_thyroid_model.pkl"
if not os.path.exists(model_path):
    train_model(X, y, model_path)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        inputs = {key: request.form[key] for key in request.form}
        row = pd.DataFrame([inputs])
        row['Age'] = float(row['Age'])  # Ensure Age is float
        prediction = predict_from_input(row, encoder, model_path)
        return render_template("index.html", prediction=prediction)
    return render_template("index.html")

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
