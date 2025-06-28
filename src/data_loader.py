import pandas as pd

def load_data(path):
    df = pd.read_csv("/home/sachingpt/thyroid/data/Thyroid_new_data.csv")
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])
    return df