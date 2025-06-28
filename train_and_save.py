from src.data_loader import load_data
from src.preprocess import preprocess_data
from src.model import train_model

df = load_data("data/Thyroid_new_data.csv")
X, y, encoder = preprocess_data(df)
train_model(X, y, "saved_model/xgboost_thyroid_model.pkl")