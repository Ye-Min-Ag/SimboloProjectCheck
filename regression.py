import streamlit as st
import pandas as pd
import joblib
from io import BytesIO
import requests

# Set title of the app
st.title('ML Prediction App')

# Upload a .xlsx file
uploaded_file = st.file_uploader('Upload an CSV file', type=['csv'])
response = requests.get("https://github.com/Ye-Min-Ag/SimboloProjectCheck/blob/main/trained_model.pkl")
content = response.content
print(type(content))
buffer = BytesIO(content)
# Load the trained model
model = joblib.load(buffer) 

# Display the prediction form if a file is uploaded
if uploaded_file is not None:
    # Read the uploaded .xlsx file
    data = pd.read_csv(uploaded_file)
    file_X = data.iloc[:,0:-1].values
    file_Y = data.iloc[:,-1].values
    # Display the uploaded data
    #st.write('Uploaded Data:')
    #st.write(data)

    # Get input features for prediction
    #input_features = data  # Adjust this based on your model's input features

    # Make predictions using the model
    predictions = model.predict(file_X)

    # Display the predictions
    st.write('Predictions:')
    st.write(predictions)
    st.write('True values:')
    st.write(file_Y)
