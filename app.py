import pandas as pd
from flask import Flask, request, jsonify
import pickle

app = Flask(__name__)

with open('heartdisease.pkl', 'rb') as heart_model_file:
    heart_model = pickle.load(heart_model_file)
    
@app.route('/', methods=['GET'])
def defaultRoute():
    return "Hello"

@app.route('/predict_heart_disease', methods=['POST'])
def predict_heart_disease():
    try:
        # Get JSON data from the request
        data = request.json  
        selected_fields = [
            'age', 'cigsPerDay', 'totChol', 'sysBP', 'diaBP', 'BMI', 'heartRate'
        ]
        
        # Loop through the data to filter the selected features only
        new_data = {field: data[field] for field in selected_fields}
        
        predicted_result = heart_model.predict([list(new_data.values())])
        
        response = {
            'predicted_result': int(predicted_result[0])
        }
        
        # Return prediction result as JSON
        return jsonify(response)  

    except Exception as e:
        error_response = {
            'error': str(e)
        }
        return jsonify(error_response), 400

# run the server
if __name__ == '__main__':
    app.run(debug=True, port=8000)
