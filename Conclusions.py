import streamlit as st

st.title("Conclusions")
st.markdown("Ensuring data quality is a fundamental step in any data science project, as it directly impacts the reliability of insights and conclusions. For our car crash dataset, we focused on evaluating and improving the accuracy, completeness, and consistency of the data to create comprehensible visualizations .")

st.header("Data Quality")
st.markdown("Large dataset, categorical data, and data only from one county")
st.markdown("Less than 5% missing data means that most of the observations or records are complete, reducing the risk of losing important information.")

st.header("Model Related Improvements")
st.markdown("Our models: KNN, Logisitic Regression, Decision Tree")

st.header("key insights:")
st.markdown("Time of day and day of the week significantly influence crash occurrence, with higher frequencies observed during peak traffic hours or weekends.")
st.markdown("This insight can inform targeted interventions like increased traffic enforcement during high-risk periods.")
st.markdown("Accidents involving pedestrians or cyclists tend to result in more severe outcomes, underscoring the importance of dedicated infrastructure for non-motorized road users.")

st.markdown("Proposed Solutions:")
st.markdown("-: Visualization: Develop intuitive dashboards or visualizations to communicate findings for people with no experience in data science")