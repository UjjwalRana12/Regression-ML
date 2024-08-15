from flask import Flask, request, jsonify
import pickle
import numpy as np
import pandas as pd
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG)

# Load the model and scaler
try:
    model = pickle.load(open('z4_heartAttack\heart_attack_model.pkl', 'rb'))
    scaler = pickle.load(open('z4_heartAttack\scaler.pkl', 'rb'))
    logging.info("Model and scaler loaded successfully.")
except Exception as e:
    logging.error(f"Error loading model or scaler: {e}")

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the JSON data from the request
        json_ = request.json
        
        # Convert the JSON data to a DataFrame
        query_df = pd.DataFrame([json_])
        
        # Debugging: Print DataFrame
        logging.info(f"Input DataFrame for prediction: {query_df}")
        
        # Check the columns and data types
        logging.info(f"Columns: {query_df.columns}")
        logging.info(f"Data types: {query_df.dtypes}")
        
        # Ensure input data has the correct order and number of features
        if query_df.shape[1] != len(scaler.mean_):  # Check if number of features matches
            raise ValueError("Number of features in input data does not match the model.")
        
        # Convert DataFrame to numpy array
        input_data_np = query_df.values
        
        # Reshape and scale the input data
        reshaped_input = input_data_np.reshape(1, -1)
        scaled_input = scaler.transform(reshaped_input)
        
        # Predict using the loaded model
        prediction = model.predict(scaled_input)
        
        # Debugging: Print the raw prediction
        logging.info(f"Raw prediction result: {prediction}")
        
        # Convert the prediction to a list (to make it JSON serializable)
        output = prediction.tolist()
        
        # Determine the prediction message
        result_message = "The person is safe from heart attack" if prediction[0] == 0 else "High risk of heart attack"
        
        return jsonify({"prediction": output, "message": result_message})
    
    except Exception as e:
        logging.error(f"Error in prediction: {e}")
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True)
