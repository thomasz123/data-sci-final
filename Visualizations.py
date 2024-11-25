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

df = pd.read_csv("weather.csv")
df = df.drop("datetime", axis = 1)
df_cleaned = df.drop("temp", axis = 1)
cols = df_cleaned.columns

st.title(":blue[Visualizations]")

tab1, tab2, tab3, tab4= st.tabs(["Histogram", "Scatterplot", "Pairplot", "Correlation Heatmap"])

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

@st.cache_data
def histogram():
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(df['temp'], kde=True)
    st.pyplot(fig)


with tab1: #histogram
    st.header("Histogram")
    histogram()
    st.markdown("This histogram shows the distribution of temperature data, and shows us the number of observations for each bin of temperatures. The distribution is made more obvious by the KDE curve. The distribution seems to be mostly symmetrical in a bell curve and unimodal, with a slight skew right.")


with tab2: #scatterplot
    st.header("Scatterplot")
    variable = st.radio("Pick one", cols)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(data = df, x = variable, y = "temp")
    st.pyplot(fig)
    st.markdown("You can choose a variable to see its scatterplot with temperature. We can see from the scatterplot if there's any semblance of correlation between the variable and temperature.")

with tab3: #pairplot
    with st.spinner("Loading visualizations..."):
        st.header('Pairplot')
        pairplot()
        st.markdown("This pairplot shows the scatterplot between any two variables. We get a more full idea of the correlations between certain variables.  The diagonal shows the countplot for that variable, and shows the distribution of the data for that variable.")


with tab4: #heatmap
    with st.spinner("Loading visualizations..."):
        st.header('Heatmap')
        heatmap()
        st.markdown("The heatmap shows the correlation value between any variables. The closer the value is to 1, the greater the correlation, and if the value is negative, the correlation is negative. We can see that dew and temperature have a correlation of 0.87, while humidiy and solar radiation have a correlation of -0.7.")


