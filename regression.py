# Import necessary libraries
import streamlit as st
# Your Machine Learning model code goes here

import pandas as pd
training_data = pd.read_csv('/content/sample_data/california_housing_train.csv')

testing_data = pd.read_csv('/content/sample_data/california_housing_test.csv')

training_data

type(training_data.shape)

print(training_data.shape[0])

training_data.describe()

training_data.dropna(axis=0, inplace=True)
testing_data.dropna(axis=0, inplace=True)

train_X = training_data.iloc[:,0:-1].values
train_Y = training_data.iloc[:,-1].values

test_X = testing_data.iloc[:,0:-1].values
test_Y = testing_data.iloc[:,-1].values

train_X
test_X

import sklearn
from sklearn.linear_model import LinearRegression
model = LinearRegression()

# fitting the data
model.fit(train_X, train_Y)    #model.fit က for modelling

test_Ypred = model.predict(test_X)

y_true = test_Y
y_pred = test_Ypred
total = 0
for i,j in zip(y_true, y_pred):
  #print('Original',i)
  #print('Predicted',j)
  #print(abs(i-j))
  total+=abs(i-j)
  #print()
print('Total',total)
print('Mean Absolute Error',total/len(y_pred))

y_true = test_Y
y_pred = test_Ypred
total = 0
for i,j in zip(y_true, y_pred):
  #print('Original',i)
  #print('Predicted',j)
  #print((i-j))
  total+=(i-j)**2
  #print()
print('Total',total)
print('Mean Square Error',total/len(y_true))

# Create a Streamlit app
st.title("Your Machine Learning Model App")
st.write("Welcome to your interactive app!")

# Add interactive components
user_input = st.text_input("Enter some text:")
st.write("You entered:", user_input)

# Display the output of your Machine Learning model
if st.button("Predict"):
    # Call your model's prediction function here
    prediction = y_pred #"This is your model's prediction"
    st.write("Prediction:", prediction)
    #print(y_true)



