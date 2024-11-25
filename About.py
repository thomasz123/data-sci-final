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

with st.spinner('Loading page...'):
    st.title(":blue[Tempature Prediction]")
    st.subheader("Weather data and analysis")

    image_path = Image.open("cloud.jpg") 
    st.image(image_path)

    st.write("Details about our project, explanation of database, etc")

    df = pd.read_csv("weather.csv")
    df = df.drop(["datetime","temp"], axis = 1)  

    st.markdown('##### WHY THIS TOPIC❓')
    st.markdown('Accurate temperature prediction is critical for weather forecasting, agriculture, and various industries. Understanding the key factors influencing temperature helps in improving prediction accuracy, enhancing decision-making for farmers, city planners, and climate researchers.')
    st.markdown("##### OUR GOAL 🎯")
    st.markdown("Our goal is to predict the temperature using multiple meteorological features and identify the most influential factors driving temperature changes. We focus on weather-related features like humidity, wind speed, pressure, and more, while aiming for an accurate forecast model.")

    st.markdown('##### OUR DATA 📊')
    st.markdown("Our dataset contains historical weather data including variables such as temperature, humidity, wind speed, and atmospheric pressure. The dataset consists of thousands of daily observations from multiple weather stations.")

    st.markdown("##### Explanation of KEY VARIABLES 📓")
    st.markdown("- :blue[Humidity]: The amount of water vapor in the air.")
    st.markdown("- :blue[Wind Speed]: Speed of the wind at a particular time.")
    st.markdown("- :blue[Pressure]: Atmospheric pressure.")
    st.markdown("- :blue[Cloud Cover]: Percentage of the sky covered by clouds.")
    st.markdown("- :blue[Precipitation]: The amount of liquid or frozen water that falls to the Earth's surface, measured over a specific period. It includes rain, snow, sleet, and hail.")
    st.markdown("- :blue[Sea Level Pressure]: The atmospheric pressure at sea level. It's a standardized measure used in meteorology to compare pressures from different elevations and is crucial for understanding weather patterns and forecasting.")
    st.markdown("- :blue[Solar Raditation]: The amount of solar energy hitting the Earth’s surface.")

    st.markdown("### Description of Data")

    st.dataframe(df.describe())
    st.markdown("This statistical description gives us more information about the count, mean, standard deviation, minimum, percentiles, and Maximum.")
    st.markdown("- :blue[Count]: All features have 1,000 data points.")
    st.markdown("- :blue[Mean]: The average value for each feature, e.g., the average humidity is 71.9969, and the average sea-level pressure is 1,016.4357.")
    st.markdown("- :blue[Standard Deviation (std)]: It indicates the spread of data around the mean. Higher values mean more variability. For example, solar radiation has a standard deviation of 45.73.")
    st.markdown("- :blue[Minimum]: The lowest recorded value in each feature. For example, the minimum wind speed is 3.5, and cloud cover is 0.")
    st.markdown("- :blue[Percentiles]: These values show the distribution of data. For example, 50 percent of humidity values are below 73.5, and 75 percent of solar radiation values are below 81.95.")
    st.markdown("- :blue[Maximum]: The highest recorded value for each feature. For example, the maximum wind speed is 25.3.")

    st.markdown("### Missing Values")
    st.markdown("Null or NaN values.")

    dfnull = df.isnull().sum()/len(df)*100
    totalmiss = dfnull.sum().round(2)
    st.write("Percentage of total missing values:",totalmiss)
    st.write(dfnull)
    if totalmiss == 0.0:
        st.success("✅ We do not exprience any missing values which is the ideal outcome of our data. We can proceed with higher accuracy in our further prediction.")
    else:
        st.warning("Poor data quality due to greater than 30 percent of missing value.")
        st.markdown(" > Theoretically, 25 to 30 percent is the maximum missing values are allowed, there's no hard and fast rule to decide this threshold. It can vary from problem to problem.")

    st.markdown("### Completeness")
    st.markdown(" The ratio of non-missing values to total records in dataset and how comprehensive the data is.")

    st.write("Total data length:", len(df))
    nonmissing = (df.notnull().sum().round(2))
    completeness= round(sum(nonmissing)/len(df),2)

    st.write("Completeness ratio:",completeness)
    st.write(nonmissing)
    if completeness >= 0.80:
        st.success("✅ We have completeness ratio greater than 0.85, which is good. It shows that the vast majority of the data is available for us to use and analyze. ")
    else:
        st.success("Poor data quality due to low completeness ratio (less than 0.85).")


