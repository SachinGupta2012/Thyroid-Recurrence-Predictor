def predict_from_input(input_df, encoder, model_path):
    import joblib
    from sklearn.preprocessing import MinMaxScaler
    import pandas as pd

    model = joblib.load(model_path)

    categorical_cols = encoder.feature_names_in_
    encoded = encoder.transform(input_df[categorical_cols])
    encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(), index=input_df.index)

    input_df = input_df.drop(columns=categorical_cols)
    scaler = MinMaxScaler()
    input_df[['Age']] = scaler.fit_transform(input_df[['Age']])

    final_df = pd.concat([input_df, encoded_df], axis=1)

    prediction = model.predict(final_df)
    return "Recurred" if prediction[0] == 1 else "Not Recurred"
