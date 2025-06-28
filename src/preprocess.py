import pandas as pd
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler

def preprocess_data(df):
    y = df['Recurred'].map({'No': 0, 'Yes': 1})
    inputs_df = df.drop('Recurred', axis=1)

    categorical_cols = inputs_df.select_dtypes(include='object').columns.tolist()

    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    encoder.fit(inputs_df[categorical_cols])

    encoded_array = encoder.transform(inputs_df[categorical_cols])
    encoded_df = pd.DataFrame(encoded_array, columns=encoder.get_feature_names_out(), index=inputs_df.index)

    final_df = pd.concat([inputs_df.drop(columns=categorical_cols), encoded_df], axis=1)

    scaler = MinMaxScaler()
    final_df[['Age']] = scaler.fit_transform(final_df[['Age']])

    return final_df, y, encoder  # ✅ Ensure three outputs
