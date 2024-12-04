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
# import folium 
# from streamlit_folium import st_folium

st.write("stuffs")
df = pd.read_csv("final_data_words.csv")
datag= pd.read_csv("gedited2.csv")
st.write("test")
df = df.drop("datetime", axis = 1)
df_cleaned = df.drop("Injury Severity", axis = 1)
cols = df_cleaned.columns
cols_uncleaned = df.columns

st.title(":red[Visualizations]")

tab1, tab2, tab3, tab4= st.tabs(["Count Plots", "Box and Whisker Plots", "Pie Charts", "Map"])

# @st.cache_data 
# def pairplot():
#     st.pyplot(sns.pairplot(df))

# @st.cache_data 
# def heatmap():
#     corr_matrix= df.corr()
#     fig, ax = plt.subplots(figsize=(10, 6))
#     sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=ax, linewidths=0.5)
#     st.pyplot(fig)
    #return fig

# @st.cache_data
# def countplot():
#     fig, ax = plt.subplots(figsize=(10, 6))
#     sns.countplot(df['Injury Severity'], kde=True)
#     st.pyplot(fig)


with tab1: #count plots
    st.header("Count Plots")

    df_count_cleaned = df.drop(columns = ["Person ID", "Circumstance", "Vehicle Year", "Vehicle Make", "Vehicle Model", "Latitude", "Longitude", "time", "date"], axis = 1)
    cols = df_count_cleaned.columns

    variable = st.radio("Pick one", cols)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.countplot(data = df, x = variable)
    plt.xticks(rotation=45, ha='right')  # Adjust rotation and horizontal alignment
    plt.tight_layout()  # Adjust layout to prevent clipping
    # plt.show()
    st.pyplot(fig)
    st.markdown("You can choose a variable to see its countplot.")


with tab2: #box and whisker plots
    st.header("Box and Whisker Plots")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(x='Injury Severity', y='Speed Limit', data=datag)
    plt.xticks(rotation=45, ha='right')  # Adjust rotation and horizontal alignment
    plt.tight_layout()  # Adjust layout to prevent clipping
    st.pyplot(fig)
    st.markdown("Injury severity as a function of speed limit")
    fig1, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(x='Injury severity cat', y='time (categorical)', data=datag)
    st.pyplot(fig1)
    st.markdown("Injury severity as a function of time")
    


with tab3: #pie charts
    st.header('Pie Charts')
    fig1, ax = plt.subplots(figsize=(10,6))
    labels = ['Mon-Fri', 'Sat-Sun']
    sizes = [75.15, 24.85]
    plt.pie(sizes,labels=labels)
    st.pyplot(fig1)
    st.markdown("Injury severity by day of the week?")

    fig, ax = plt.subplots(figsize=(10,6))
    labels = ['Mon-Fri', 'Sat-Sun']
    sizes = [54.74, 45.26]
    plt.pie(sizes,labels=labels)
    st.pyplot(fig)
    st.markdown("...")

with tab4: #map
    st.header('Map')

    image_path = Image.open("map.png") 
    st.image(image_path)

    # map_center = [df["Latitude"].mean(), df["Longitude"].mean()]
    # mymap = folium.Map(location = map_center, zoom_start = 8)

    # for index, row in df.iterrows():
    #     if row['Latitude'] and row['Longitude']:
    #         folium.Marker([row['Latitude'], row['Longitude']], popup = row['Report Number']).add_to(mymap)

    # st_map = st_folium(mymap, width = 700, height = 450)
    st.markdown("This map shows the location of each of the accidents in our dataset.")

