import requests
import json

# Replace with your Cloud Function's URL
url = "https://europe-west3-mlops-401018.cloudfunctions.net/sklearn"

# Input data
input_data = "1,2,3,4,5"  # Modify this with your actual input data

# Create a JSON payload
payload = {"input_data": input_data}

# Send a POST request
response = requests.post(url, json=payload)

# Check the response
if response.status_code == 200:
    prediction = response.text
    print(f"Prediction: {prediction}")
else:
    print(f"Request failed with status code {response.status_code}: {response.text}")
