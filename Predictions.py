import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
from PIL import Image
import codecs
import streamlit.components.v1 as components
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn import metrics
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import matplotlib.pyplot as plt

st.title(":blue[Predictions]")

tab1, tab2 = st.tabs(["Linear Regression", "Logistic Regression"])
df = pd.read_csv("weather.csv")
df = df.drop(["datetime"], axis = 1)  

with tab1:
    df2 = df.drop("temp", axis = 1)
    columns = df2.columns
    input = st.multiselect("Select variables:",columns,["dew"])
    df2 = df2[input]
    
    X = df2
    y = df["temp"]

    if input == []:
        st.toast("Please Choose a Variable")

    else:
        X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2)
        lr = LinearRegression()
        lr.fit(X_train, y_train)
        prediction = lr.predict(X_test)

        #Print MAE and R2 values
        mae = metrics.mean_absolute_error(prediction, y_test)
        st.write("Mean Absolute Error:", mae)
        r2 = metrics.r2_score(prediction,y_test)
        st.write("R2:", r2)

        #Linear Regression
        with st.spinner('Loading scatterplot...'):
            fig, ax = plt.subplots(figsize = (10,6))
            plt.title("Actual vs Predicted Temperatures",fontsize=25)
            plt.xlabel("Actual Temperatures",fontsize=18)
            plt.ylabel("Predicted Temperatures", fontsize=18)
            plt.scatter(x=y_test,y=prediction)

            st.pyplot(fig)




    # #Linear Regression (Temp vs. Dew)
    # st.title("Predicting Temperature based on Dew")
    # list_columns = df.columns

    # # Step 1: Splitting the dataset into X and y
    # X1 = df[["dew"]]  # Use double brackets to make X1 a 2D array
    # y1 = df["temp"]

    # # Step 2: Splitting into 4 chunks: X_train, X_test, y_train, y_test
    # X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, test_size=0.2)

    # # Step 3: Initialize the LinearRegression model
    # lr = LinearRegression()

    # # Step 4: Train the model
    # lr.fit(X1_train, y1_train)

    # # Step 5: Predictions
    # predictions = lr.predict(X1_test)

    # # Step 6: Plot the results
    # with st.spinner('Loading scatterplot...'):
    #     # fig, ax = plt.subplots(figsize=(10, 6))
    #     # plt.title("Temperature and Dew", fontsize=25)
    #     # plt.xlabel("Dew", fontsize=18)
    #     # plt.ylabel("Temp", fontsize=18)

    #     # # Correct the axes for the scatter plot
    #     # plt.scatter(x=X1_test, y=y1_test, color='blue', label='Actual')
    #     # #plt.scatter(x=X1_test, y=predictions, color='red', label='Predicted')
    #     # plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linewidth=2)     
    #     # plt.legend()


    #     plt.figure(figsize=(8, 6))
    #     plt.scatter(y_test, predictions)
    #     plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linewidth=2)
    #     plt.xlabel('Actual CO2 Emissions')
    #     plt.ylabel('Predicted CO2 Emissions')
    #     plt.title('Actual vs Predicted CO2 Emissions')
    #     plt.show()
        
    #     st.pyplot(fig)

    # #Ranking variables
    # # coeff_df = pd.DataFrame(lr.coef_, X.columns, columns=['Coefficient'])
    # # coeff_df
    # # feature_names = [f'Feature_{i}' for i in list(X.columns)]
    # # df_X = pd.DataFrame(X, columns=feature_names)
    # # # Coefficients represent the importance in linear regression
    # # coefficients = lr.coef_

    # # # Making the coefficients positive to compare magnitude
    # # importance = np.abs(coefficients)

    # # # Plotting feature importance with feature names
    # # plt.figure(figsize=(10, 8))
    # # plt.barh(feature_names, importance)
    # # plt.xlabel('Absolute Coefficient Value')
    # # plt.title('Feature Importance (Linear Regression)')
    # # plt.show()


    # # #Stp6 Evaluation

    # mae=metrics.mean_absolute_error(predictions,y1_test)
    # r2=metrics.r2_score(predictions,y1_test)

    # st.write("Mean Absolute Error:",mae)
    # st.write("R2 output:",r2)

    # #Logistic Regression

with tab2: 
    df_logistic = df
    df_logistic['precipitation'] = df_logistic['precip'].apply(lambda x: 1 if x > 0 else 0)
    
    df_logistic2 = df_logistic.drop(["precipitation", "precip"], axis = 1)

    st.dataframe(df_logistic2)

    columns = df_logistic2.columns
    loginput = st.multiselect("Select variables:",columns,["dew"])

    df_logistic2 = df_logistic[input]
    
    #st.pyplot create a countplot to count the number of rainy and non rainy days

    Xlog = df_logistic2
    ylog = df_logistic["precipitation"]

    Xlog_train, Xlog_test, ylog_train, ylog_test = train_test_split(Xlog,ylog,test_size = 0.2)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(Xlog_train)
    X_test_scaled = scaler.transform(Xlog_test)

    logmodel = LogisticRegression()
    logmodel.fit(X_train_scaled, ylog_train)
    logprediction = logmodel.predict(X_test_scaled)
    

    # Create confusion matrix for plotting the comparison between true labels and predictions
    cm = confusion_matrix(ylog_test, logprediction)
    
    fig, ax = plt.subplots(figsize = (10,6))
    sns.heatmap(pd.DataFrame(cm), annot = True, cmap = "YlGnBu")
    plt.title("Confusion matrix",fontsize=25)
    plt.xlabel("Predicted",fontsize=18)
    plt.ylabel("Actual", fontsize=18)
    plt.scatter(x=ylog_test,y=logprediction)
    st.pyplot(fig)

    st.write("Accuracy:", accuracy_score(ylog_test, logprediction) * 100, "%")

    # Create a barplot comparing actual 0s and 1s vs predicted 0s and 1s
    true_counts = pd.Series(ylog_test).value_counts().sort_index()
    pred_counts = pd.Series(logprediction).value_counts().sort_index()

    # Aligning the series for 0s and 1s to have the same indexes
    true_counts = true_counts.reindex([0, 1], fill_value=0)
    pred_counts = pred_counts.reindex([0, 1], fill_value=0)

    # Plotting
    labels = ['0', '1']
    x = np.arange(len(labels))  # the label locations

    fig, ax = plt.subplots(figsize=(8, 6))
    width = 0.35  # the width of the bars

    # Plot the bars
    rects1 = ax.bar(x - width/2, true_counts, width, label='True')
    rects2 = ax.bar(x + width/2, pred_counts, width, label='Predicted')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_xlabel('Class')
    ax.set_ylabel('Count')
    ax.set_title('Comparison of True vs Predicted Values for Logistic Regression')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    # Display the plot
    st.pyplot(fig)

