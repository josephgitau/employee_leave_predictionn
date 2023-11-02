## Streamlit app for predicting if a person will be on leave or not

# Importing libraries
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import classification_report
import pickle

# ignore warnings
import warnings
warnings.filterwarnings('ignore')

# App title
st.title('Employee Leave Prediction')

# add an upload button for data upload
uploaded_file = st.file_uploader("Upload your input CSV file", 
                                 type=["csv"])

# create a dataframe from the uploaded file
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    # get input from user on the number of rows to display
    num_rows = st.number_input('Enter the number of rows to display', 
                            min_value=0, max_value=30, value=5)
    # show the top 5 row of the dataframe
    st.header("Data Sample")
    st.dataframe(data.head(num_rows))


# create a function to plot categorical variables
def plot_cat(data, cat_var):
    st.header("Plot of " + cat_var)
    fig, ax = plt.subplots()
    sns.set_style('darkgrid')
    sns.countplot(data=data, x=cat_var)
    plt.title(cat_var)
    plt.show()
    st.pyplot(fig)

# get a list of all the columns in the dataframe
columns = data.columns.tolist()

# create a dropdown where user can select the column to plot
cat_var = st.selectbox('Select a column to plot', columns)

# plot the selected column
plot_cat(data, cat_var)

# create a function to encode categorical variables
def encode_cat(data, cat_var):
    encoder = OrdinalEncoder()
    data[cat_var] = encoder.fit_transform(data[[cat_var]])
    return data

# for loop to encode all categorical variables
for i in data.columns:
    if data[i].dtypes == 'object':
        encode_cat(data, i)

# show the top 3 our updated dataframe
st.header("Data Encoded Dataframe Sample")
st.dataframe(data.head(3))

# Create our target and features
X = data.drop(columns=['LeaveOrNot'])

# import the model
model = pickle.load(open('model.pkl', 'rb'))

# make predictions using the model
prediction = model.predict(X)

# add the predictions to the dataframe
data['LeaveOrNot_prediction'] = prediction

# get user input on the number of rows to display
num_rows_pred = st.number_input('Enter the number of rows to display', 
                            min_value=0, max_value=50, value=5)

# show the top 5 rows of the dataframe
st.header("Predictions")
st.dataframe(data.head(num_rows_pred))

# print the classification report
st.header("Classification Report")
st.text("0 = Will not be on leave, 1 = Will be on leave")

class_report = classification_report(data['LeaveOrNot'],
                                    data['LeaveOrNot_prediction'])
st.text(class_report)
