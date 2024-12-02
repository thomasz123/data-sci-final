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
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

st.title(":red[Predictions]")

tab1, tab2, tab3 = st.tabs(["Logisitic Regression", "KNN Classifier", "Decision Tree"])
df = pd.read_csv("mynewdata.csv")
#df = df.drop(["datetime"], axis = 1)  

with tab1:
    st.header("logistic")
    df_logistic = df
    df_logistic['Injury'] = df_logistic['Injury Severity'].apply(lambda x: 0 if x == 1 else 1)
    
    df_logistic2 = df_logistic.drop(["day of the week", "date", "time", "Report Number", "Circumstance"], axis = 1)

    # st.dataframe(df_logistic)

    columns = df_logistic2.columns
    loginput = st.multiselect("Select variables:",columns,["Weather"])

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
    st.header("knn")
    ## divide the dataset in 4 chuncks X train X test y train and y test
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)

    # Standardize the feature values
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X_train,y_train)

    knn.score(X_test,y_test)
    results = knn.predict(X_test)
    results

with tab3: 
    st.header("decision tree")
    
    #
    
    # Create Decision Tree classifer object
    clf = DecisionTreeClassifier()

    # Train Decision Tree Classifer
    clf = clf.fit(X_train,y_train)

    #Predict the response for test dataset
    y_pred = clf.predict(X_test)
    # Model Accuracy, how often is the classifier correct?
    print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

    feature_cols = X.columns
    feature_cols

    import graphviz
    from sklearn.tree import export_graphviz
    feature_names = X.columns
    dot_data = export_graphviz(clf, out_file=None,

                            feature_names=feature_cols,

                            class_names=['0','1'],

                            filled=True, rounded=True,

                            special_characters=True)

    graph = graphviz.Source(dot_data)
    graph
