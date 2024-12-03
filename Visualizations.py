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

df = pd.read_csv("final_data_words.csv")
#df = df.drop("datetime", axis = 1)
df_cleaned = df.drop("Injury Severity", axis = 1)
cols = df_cleaned.columns
cols_uncleaned = df.columns

st.title(":red[Visualizations]")

tab1, tab2, tab3, tab4= st.tabs(["Count Plots", "Box and Whisker Plots", "Pie Charts", "Map"])

@st.cache_data 
def pairplot():
    st.pyplot(sns.pairplot(df))

@st.cache_data 
def heatmap():
    corr_matrix= df.corr()
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=ax, linewidths=0.5)
    st.pyplot(fig)
    #return fig

# @st.cache_data
# def countplot():
#     fig, ax = plt.subplots(figsize=(10, 6))
#     sns.countplot(df['Injury Severity'], kde=True)
#     st.pyplot(fig)


with tab1: #count plots
    st.header("Count Plots")
    variable = st.radio("Pick one", cols_uncleaned)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.countplot(data = df, x = variable)
    st.markdown("You can choose a variable to see its countplot.")


with tab2: #box and whisker plots
    st.header("Scatterplot")
    variable = st.radio("Pick one", cols)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(data = df, x = variable, y = "Injury Severity")
    st.pyplot(fig)
    st.markdown("You can choose a variable to see its scatterplot with temperature. We can see from the scatterplot if there's any semblance of correlation between the variable and temperature.")

with tab3: #pie charts
    with st.spinner("Loading visualizations..."):
        st.header('Pairplot')
        pairplot()
        st.markdown("This pairplot shows the scatterplot between any two variables. We get a more full idea of the correlations between certain variables.  The diagonal shows the countplot for that variable, and shows the distribution of the data for that variable.")


with tab4: #map
    with st.spinner("Loading visualizations..."):
        st.header('Heatmap')
        heatmap()
        st.markdown("The heatmap shows the correlation value between any variables. The closer the value is to 1, the greater the correlation, and if the value is negative, the correlation is negative. We can see that dew and temperature have a correlation of 0.87, while humidiy and solar radiation have a correlation of -0.7.")


