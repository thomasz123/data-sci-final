import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
from PIL import Image
import codecs
import streamlit.components.v1 as components
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn import metrics
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import matplotlib.pyplot as plt
import graphviz
from sklearn.neighbors import KNeighborsClassifier

st.title(":red[Predictions]")

tab1, tab2, tab3 = st.tabs(["Logisitic Regression", "KNN", "Decision Tree"])
df = pd.read_csv("mynewdata.csv")

#df = df.drop(["datetime"], axis = 1)  

with tab1:
    st.header("logistic")
    df_logistic = df
    df_logistic['Injury'] = df_logistic['Injury Severity'].apply(lambda x: 0 if x == 1 else 1)
    
    df_logistic2 = df_logistic.drop(["day of the week", "date", "time", "Report Number", "Circumstance", "Injury Severity", "Injury"], axis = 1)

    df_logistic2
    # st.dataframe(df_logistic)

    logcolumns = df_logistic2.columns
    loginput = st.multiselect("Select variables:",logcolumns,["Weather"])

    df_logistic2 = df_logistic[loginput]
    
    # #st.pyplot create a countplot to count the number of rainy and non rainy days

    Xlog = df_logistic2
    ylog = df_logistic["Injury"]

    Xlog_train, Xlog_test, ylog_train, ylog_test = train_test_split(Xlog,ylog,test_size = 0.2)

    # scaler = StandardScaler()
    # X_train_scaled = scaler.fit_transform(Xlog_train)
    # X_test_scaled = scaler.transform(Xlog_test)

    logmodel = LogisticRegression()
    logmodel.fit(Xlog_train, ylog_train)
    logprediction = logmodel.predict(Xlog_test)
    

    # # Create confusion matrix for plotting the comparison between true labels and predictions
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

with tab2: 
    st.header("KNN")
    
    df_knn = df
    df_knn['Injury'] = df_knn['Injury Severity'].apply(lambda x: 0 if x == 1 else 1)
    
    df_knn2 = df_knn.drop(["day of the week", "date", "time", "Report Number", "Circumstance", "Injury Severity", "Injury"], axis = 1)

    # st.dataframe(df_logistic)

    knncolumns = df_knn2.columns
    knninput = st.multiselect("Select variables:", knncolumns, ["Light"])

    df_knn2 = df_knn[knninput]
    
    # #st.pyplot create a countplot to count the number of rainy and non rainy days

    Xknn = df_knn2
    yknn = df_knn["Injury"]

    Xknn_train, Xknn_test, yknn_train, yknn_test = train_test_split(Xknn,yknn,test_size = 0.3)

    # Standardize the feature values
    scaler = StandardScaler()
    Xknn_train = scaler.fit_transform(Xknn_train)
    Xknn_test = scaler.transform(Xknn_test)

    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(Xknn_train,yknn_train)

    st.write("Accuracy:", knn.score(Xknn_test,yknn_test) * 100, "%")
    results = knn.predict(Xknn_test)
    # st.write(results)

    k_list = range(1, 31)
    k_values = dict(n_neighbors=k_list)

    grid = GridSearchCV(knn, k_values, cv=5, scoring='accuracy')
    grid.fit(df_knn2, df_knn['Injury'])

    # for key in grid.cv_results_.keys():
    #     st.write(key)
    
    st.write("The best value of k = {} with {} of accuracy.".format(grid.best_params_,grid.best_score_))
    st.write("The best classifier is: {}".format(grid.best_estimator_))

    graphic = grid.cv_results_['mean_test_score']
    graphic

    fig, ax = plt.subplots(figsize=(10, 6))
    plt.plot(k_list,graphic,color='navy',linestyle='dashed',marker='o')
    plt.xlabel('K Number of Neighbors', fontdict={'fontsize': 15})
    plt.ylabel('Accuracy', fontdict={'fontsize': 15})
    plt.title('K NUMBER X ACCURACY', fontdict={'fontsize': 30})
    plt.xticks(range(0,31,3),)
    #plt.xaxis.set_major_locator(MultipleLocator(3))
    st.pyplot(fig)

with tab3: 
    st.header("Decision Tree")
    
    df_dt = df
    df_dt['Injury'] = df_dt['Injury Severity'].apply(lambda x: 0 if x == 1 else 1)
    
    df_dt2 = df_dt.drop(["day of the week", "date", "time", "Report Number", "Circumstance", "Injury Severity", "Injury"], axis = 1)

    # st.dataframe(df_logistic)

    dtcolumns = df_dt2.columns
    dtinput = st.multiselect("Select variables:", dtcolumns, ["Collision Type"])

    df_dt2 = df_dt[dtinput]
    
    # #st.pyplot create a countplot to count the number of rainy and non rainy days

    Xdt = df_dt2
    ydt = df_dt["Injury"]

    Xdt_train, Xdt_test, ydt_train, ydt_test = train_test_split(Xdt,ydt,test_size = 0.3)

    # Standardize the feature values
    scaler = StandardScaler()
    Xdt_train = scaler.fit_transform(Xdt_train)
    Xdt_test = scaler.transform(Xdt_test)
    
    #Create Decision Tree classifer object
    clf = DecisionTreeClassifier()

    # Train Decision Tree Classifer
    clf = clf.fit(Xdt_train,ydt_train)

    # Predict the response for test dataset
    ydt_pred = clf.predict(Xdt_test)
    # # Model Accuracy, how often is the classifier correct?
    st.write("Accuracy:",metrics.accuracy_score(ydt_test, ydt_pred)*100, "%")

    
    feature_cols = Xdt.columns
    dot_data = export_graphviz(clf, out_file=None,

                            feature_names=feature_cols,

                            class_names=['0','1'],

                            filled=True, rounded=True,

                            special_characters=True)

    graph = graphviz.Source(dot_data)
    graph  
