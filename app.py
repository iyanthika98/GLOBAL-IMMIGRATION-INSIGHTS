from flask import Flask, request, jsonify
import pandas as pd
import joblib
import traceback

app = Flask(__name__)

numerical_cols = ['Value', 'Population', 'hci_Rank', 'hci_index', 'hdi_index', 'migration_country_population']
categorical_cols = ['Country_code', 'nationality_country', 'Variable', 'Year_x', 'migration_country_population_bins']

def dense_transform(X):
    if hasattr(X, 'toarray'):
        return X.toarray()
    return X

@app.route('/', methods=['GET', 'POST'])
def predict():
    data = None
    if request.method == 'POST':
        try:
            data = {col: request.form.get(col, '') for col in numerical_cols + categorical_cols}
            data_df = pd.DataFrame([data])
            data_df[numerical_cols] = data_df[numerical_cols].apply(pd.to_numeric, errors='coerce')

            naive_bayes_model = joblib.load(r'D:\SEMESTER_4\deploy_all\naive_bayes_model.pkl')
            label_encoder = joblib.load(r'D:\SEMESTER_4\deploy_all\label_encoder.pkl')
            
            prediction = naive_bayes_model.predict(data_df)
            predicted_countries = label_encoder.inverse_transform(prediction)

            return jsonify({'prediction': predicted_countries.tolist()})
        except Exception as e:
            error_message = f"An unexpected error occurred: {str(e)}"
            traceback_details = traceback.format_exc()
            return jsonify({'error': error_message, 'traceback': traceback_details, 'input_data': data})

    return '''
    <form method="post">
    <div><input type="text" name="Value" placeholder="Value"></div>
    <div><input type="text" name="Population" placeholder="Population"></div>
    <div><input type="text" name="hci_Rank" placeholder="hci_Rank"></div>
    <div><input type="text" name="hci_index" placeholder="hci_index"></div>
    <div><input type="text" name="hdi_index" placeholder="hdi_index"></div>
    <div><input type="text" name="migration_country_population" placeholder="migration_country_population"></div>
    <div><input type="text" name="Country_code" placeholder="Country_code"></div>
    <div><input type="text" name="nationality_country" placeholder="Nationality_country"></div>
    <div><input type="text" name="Variable" placeholder="Variable"></div>
    <div><input type="text" name="Year_x" placeholder="Year_x"></div>
    <div><input type="text" name="migration_country_population_bins" placeholder="Migration_country_population_bins"></div>
    <div><input type="submit" value="Predict"></div>
    </form>
    '''

if __name__ == '__main__':
    app.run(debug=True, port=5000)
