import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
from PIL import Image
import codecs
import streamlit.components.v1 as components

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

import matplotlib.pyplot as plt

st.title("Data Science App")


app_page = st.sidebar.selectbox("Select Page", ['Data Exploration', 'Visualization', 'Prediction'])
df = pd.read_csv("weather.csv")

df = df.drop("datetime", axis = 1)  

df2 = df.drop("temp", axis = 1)


if app_page == 'Data Exploration':
    st.header("Data Exploration")
    image_path = Image.open("cloud.jpg") 
    st.image(image_path)

if app_page == 'Visualization':
    st.header("Visualization")
    st.pyplot(sns.pairplot(df))

    #heat Map
    corr_matrix= df.corr()
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=ax, linewidths=0.5)
    st.pyplot(fig)

if app_page == 'Prediction':
    st.header("Prediction")
    X = df2
    y = df["temp"]

    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2)
    lr = LinearRegression()

    lr.fit(X_train, y_train)

    prediction = lr.predict(X_test)

    mae = metrics.mean_absolute_error(prediction, y_test)
    st.write("Mean Absolute Error:", mae)

    r2 = metrics.r2_score(prediction,y_test)
    st.write("R2:", r2)


