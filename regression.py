'''import streamlit as st
import pandas as pd
import pickle
from io import BytesIO
import requests

st.title('ML Prediction App')
uploaded_file = st.file_uploader('Upload an CSV file', type=['csv'])
response = requests.get("https://github.com/Ye-Min-Ag/SimboloProjectCheck/blob/main/trained_model.pkl")
model_content = response.content
model = pickle.loads(model_content)
if uploaded_file is not None:
    # Read the uploaded .xlsx file
    data = pd.read_csv(uploaded_file)
    file_X = data.iloc[:,0:-1].values
    file_Y = data.iloc[:,-1].values
    predictions = model.predict(file_X)
    # Display the predictions
    st.write('Predictions:')
    st.write(predictions)
    st.write('True values:')
    st.write(file_Y)'''
import streamlit as st
import pandas as pd
import pickle
model = pickle.load(open('trained_model.pkl','rb'))
# Set title of the app
st.set_page_config(
    page_title="Prediction App",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="expanded",
)

def predict_value(f):
    input = f
    prediction = model.predict(input)
    return prediction

def main():
    # Upload a .csv file
    uploaded_file = st.file_uploader('Upload a CSV file', type=['csv'])
    if uploaded_file is not None:
        feature = feature_extraction(uploaded_file)
        Predicted_values = predict_value(feature)
    st.write('Predictions:')
    st.write(Predicted_values)

# Display the prediction form if a file is uploaded
def feature_extraction(file):
    # Read the uploaded .csv file
    data = pd.read_csv(file)
    file_X = data.iloc[:, 0:-1].values
    file_Y = data.iloc[:, -1].values
    return file_X

if __name__=='__main__':
    main()


