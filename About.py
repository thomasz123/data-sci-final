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
    st.title(":red[Comprehensive Analysis of Car Crash Factors]")
    st.subheader("Exploring the Impact of Different Variables on Road Safety")

    image_path = Image.open("Car.jpeg") 
    st.image(image_path)

    st.write("Details about our project, explanation of database, etc")

    df = pd.read_csv("mynewdata.csv")
    #df = df.drop(["datetime","temp"], axis = 1)  

    st.markdown('##### WHY THIS TOPIC? 🔍')
    st.markdown('Understanding the factors contributing to car crashes is essential for improving road safety, urban planning, and emergency response systems. By analyzing patterns in crash data, we can identify high-risk areas, timeframes, and behaviors, enabling proactive measures to reduce accidents and save lives.')
    st.markdown("##### OUR GOAL 🎯")
    st.markdown("Our goal is to analyze car crash data to uncover the key factors influencing collisions. Using variables like crash time, road conditions, and driver behavior, we aim to develop insights that inform safety policies and predict potential risks. This will help improve preventative measures, enforcement, and urban traffic management.")

    st.markdown('##### OUR DATA 📂')
    st.markdown("Our dataset includes comprehensive information on car crashes, with variables such as:")
    st.markdown(":red[Crash Date/Time]: When the crash occurred.")
    st.markdown(":red[Location Details]: Road name, cross-street name, and municipality.")
    st.markdown(":red[Environmental Conditions:]: Weather, surface condition, light condition.")
    st.markdown(":red[Driver Information:]: Distractions, substance abuse, and fault determination.")
    st.markdown(":red[Vehicle Data:]: Vehicle type, movement, speed limit, and damage extent.")


    st.markdown("### Description of Data🧾")

    st.dataframe(df.info())
    st.markdown("This statistical description gives us more information about the count, mean, standard deviation, minimum, percentiles, and Maximum.")
    st.markdown(":red[Count]: All features have 2,000 data points ensuring a sufficient sample size for analysis.")
    st.markdown(":red[Mean]: The average value for each feature, for example, the average crash severity index is")
    st.markdown(":red[Standard Deviation (std)]: It indicates the spread of data around the mean. Higher values mean more variability. For example, solar radiation has a standard deviation of 45.73.")
    st.markdown(":red[Minimum]: The lowest recorded value in each feature. For example, the minimum wind speed is 3.5, and cloud cover is 0.")
    st.markdown(":red[Percentiles]: These values show the distribution of data. For example, 50 percent of humidity values are below 73.5, and 75 percent of solar radiation values are below 81.95.")
    st.markdown(":red[Maximum]: The highest recorded value for each feature. For example, the maximum wind speed is 25.3.")

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


