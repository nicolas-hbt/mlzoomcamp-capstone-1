import pickle
import pandas as pd
import train  # Importing to use the clean_dataframe function
from flask import Flask, request, jsonify

# Define the model file path
MODEL_FILE = 'model.bin'

# Initialize Flask App
app = Flask("job_change_prediction")

# Load the trained model pipeline (includes preprocessing and classifier)
print(f"Loading model from {MODEL_FILE}...")
with open(MODEL_FILE, 'rb') as f_in:
    model_pipeline = pickle.load(f_in)


@app.route('/predict', methods=['POST'])
def predict():
    """
    Accepts candidate data as JSON, cleans it, and returns
    the probability of them looking for a job change.
    """
    try:
        # 1. Get JSON data from the request
        candidate_data = request.get_json()

        # 2. Convert to DataFrame (List of dicts approach)
        # This handles both single instances and batch requests if needed
        X_raw = pd.DataFrame([candidate_data])

        # 3. Apply the manual cleaning step from train.py
        # (Converts ordinal strings like '>20' to numbers, etc.)
        X_clean = train.clean_dataframe(X_raw)

        # 4. Generate Prediction
        # The pipeline handles imputation, encoding, and scaling automatically
        # We predict_proba to get the probability of class 1 (Looking for job)
        y_pred_prob = model_pipeline.predict_proba(X_clean)[:, 1]

        # Get the actual class (0 or 1) based on default 0.5 threshold
        y_pred_class = model_pipeline.predict(X_clean)

        # 5. Prepare response
        result = {
            'job_change_probability': float(y_pred_prob[0]),
            'looking_for_job': bool(y_pred_class[0])
        }

        return jsonify(result)

    except Exception as e:
        # Return error message if something fails (e.g., malformed data)
        return jsonify({'error': str(e)}), 400


if __name__ == '__main__':
    # Run the server
    app.run(debug=True, host='0.0.0.0', port=9696)