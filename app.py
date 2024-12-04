import streamlit as st
from PIL import Image
import codecs
import streamlit.components.v1 as components

#page navigations

about = st.Page("About.py", title="About", icon="📝")
visualizations = st.Page("Visualizations.py", title="Visualizations ", icon="📊")
predictions = st.Page("Predictions.py", title="Predictions", icon="🤖")
conclusions = st.Page("Conclusions.py", title="Conclusions")

pg = st.navigation([about, visualizations, predictions, conclusions])
#st.set_page_config(page_title="About", page_icon="📝")
pg.run()