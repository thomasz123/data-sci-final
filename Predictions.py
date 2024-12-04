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
from shapash.explainer.smart_explainer import SmartExplainer
import random
import mlflow
import dagshub
from mlflow import log_metric, log_param, log_artifact

#dagshub.init(repo_owner='thomasz123', repo_name='data-sci-final', mlflow=True)

st.title(":red[Predictions]")

tab1, tab2, tab3, tab4, tab5 = st.tabs(["Logisitic Regression", "KNN", "Decision Tree", "Explainable AI", "MLFlow"])
df = pd.read_csv("mynewdata.csv")

#df = df.drop(["datetime"], axis = 1)  

with tab1:
    st.header("logistic")
    df_logistic = df
    df_logistic['Injury'] = df_logistic['Injury Severity'].apply(lambda x: 0 if x == 1 else 1)
    
    df_logistic2 = df_logistic.drop(["day of the week", "date", "time", "Report Number", "Circumstance", "Injury Severity", "Injury"], axis = 1)

    # df_logistic2
    # st.dataframe(df_logistic)

    logcolumns = df_logistic2.columns

    test = st.multiselect("Select variables:",logcolumns,["Weather"], key = 1)
    everything = st.checkbox("Choose all variables", key = 4)

    if everything:
        loginput = logcolumns
    else:
        loginput = test

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
    sns.heatmap(pd.DataFrame(cm), annot = True, fmt=".0f", cmap = "YlGnBu")
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
    knninput = st.multiselect("Select variables:", knncolumns, ["Weather"], key = 2)

    dteverything = st.checkbox("Choose all variables", key = 5)

    if dteverything:
        knninput = knncolumns
        
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

    # st.write("Accuracy:", knn.score(Xknn_test,yknn_test) * 100, "%")
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
    # graphic

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
    dtinput = st.multiselect("Select variables:", dtcolumns, ["Weather"], key = 3)
    dteverything = st.checkbox("Choose all variables", key = 6)

    if dteverything:
        dtinput = dtcolumns
    
    # #st.pyplot create a countplot to count the number of rainy and non rainy days

    df_dt2 = df_dt[dtinput]

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

with tab4: 
    st.header("Explainable AI")
    xpl = SmartExplainer(clf)
    y_pred = pd.Series(logprediction)
    X_test = Xlog_test.reset_index(drop=True)
    xpl.compile(x=X_test, y_pred=y_pred)
    # fig, ax = plt.subplots(figsize = (10,6))
    
    st.write(xpl.plot.features_importance())

    subset = random.choices(X_test.index, k =50)
    st.write(xpl.plot.features_importance(selection=subset))

    df_cols = df_logistic2.columns
    choice = st.radio("Pick a variable", df_cols)

    st.write(xpl.plot.contribution_plot(choice))


with tab5:
    st.header("Hyperparameter Tuning")
    st.link_button("Go to MLFlow", "https://dagshub.com/thomasz123/data-sci-final/experiments")

    dt_train = st.button("Train Decision Tree")

    if dt_train:

        with mlflow.start_run():

            # Split the data into training and testing sets
            X_train, X_test, y_train, y_test = train_test_split(Xdt, ydt, test_size=0.3, random_state=42)

            # Create a decision tree classifier
            dt = DecisionTreeClassifier(random_state=42)

            # Define a parameter grid to search over
            param_grid = {'max_depth': [3, 5, 10], 'min_samples_leaf': [1, 2, 4]}

            # Create GridSearchCV object
            grid_search = GridSearchCV(estimator=dt, param_grid=param_grid, cv=5)

            # Perform grid search to find the best parameters
            grid_search.fit(X_train, y_train)

            # Log the best parameters
            best_params = grid_search.best_params_
            mlflow.log_params(best_params)

            #y_pred 

            # Evaluate the model
            best_dt = grid_search.best_estimator_
            test_score = best_dt.score(X_test, y_test)

            # Log the performance metric
            accuracy = metrics.accuracy_score(y_test, ydt_pred)
            precision = metrics.precision_score(y_test, ydt_pred)
            recall = metrics.recall_score(y_test, ydt_pred)
            f1 = metrics.f1_score(y_test, ydt_pred)

            log_metric("accuracy", accuracy)
            log_metric("precision", precision)
            log_metric("recall", recall)
            log_metric("f1", f1)


            print(f"Accuracy: {accuracy}")
            print(f"Precision: {precision}")
            print(f"Recall: {recall}")
            print(f"F1 Score: {f1}")
            # Log the best model in MLflow
            mlflow.sklearn.log_model(best_dt, "best_dt")

            # Save the model to the MLflow artifact store
            mlflow.sklearn.save_model(best_dt, "best_dt_model")

            mlflow.end_run()

    log_train = st.button("Train Logisitic Regression")

    if log_train:

        with mlflow.start_run():

            # Split the data into training and testing sets
            X_train, X_test, y_train, y_test = train_test_split(Xlog, ylog, test_size=0.2, random_state=42)

            lr_param_grid = {'C': [0.1, 1, 10]}
            lr = LogisticRegression(random_state=42)

    
            # Create GridSearchCV object
            grid_search = GridSearchCV(estimator=lr, param_grid=lr_param_grid, cv=5)

            # Perform grid search to find the best parameters
            grid_search.fit(X_train, y_train)

            # Log the best parameters
            best_params = grid_search.best_params_
            mlflow.log_params(best_params)

            #y_pred 

            # Evaluate the model
            best_lr = grid_search.best_estimator_
            test_score = best_lr.score(X_test, y_test)

            # Log the performance metric
            accuracy = metrics.accuracy_score(y_test, logprediction)
            precision = metrics.precision_score(y_test, logprediction)
            recall = metrics.recall_score(y_test, logprediction)
            f1 = metrics.f1_score(y_test, logprediction)

            log_metric("accuracy", accuracy)
            log_metric("precision", precision)
            log_metric("recall", recall)
            log_metric("f1", f1)


            print(f"Accuracy: {accuracy}")
            print(f"Precision: {precision}")
            print(f"Recall: {recall}")
            print(f"F1 Score: {f1}")
            # Log the best model in MLflow
            mlflow.sklearn.log_model(best_lr, "best_lr")

            # Save the model to the MLflow artifact store
            mlflow.sklearn.save_model(best_lr, "best_lr_model")

            mlflow.end_run()

    knn_train = st.button("Train KNN")

    if knn_train:

        with mlflow.start_run():

            # Split the data into training and testing sets
            X_train, X_test, y_train, y_test = train_test_split(Xknn, yknn, test_size=0.3, random_state=42)

            # Create KNN
            knn = KNeighborsClassifier(n_neighbors=3)

            # Define a parameter grid to search over
            # param_grid = {'max_depth': [3, 5, 10], 'min_samples_leaf': [1, 2, 4]}

            # Create GridSearchCV object
            knn = KNeighborsClassifier(n_neighbors=3)
            knn.fit(Xknn_train,yknn_train)

            # st.write("Accuracy:", knn.score(Xknn_test,yknn_test) * 100, "%")
            results = knn.predict(Xknn_test)
            # st.write(results)

            k_list = range(1, 31)
            k_values = dict(n_neighbors=k_list)

            grid = GridSearchCV(knn, k_values, cv=5, scoring='accuracy')
            grid.fit(df_knn2, df_knn['Injury'])

            # Log the best parameters
            knnbest_params = grid.best_params_
            mlflow.log_params(knnbest_params)

            #y_pred 

            # Evaluate the model
            best_knn = grid.best_estimator_
            test_score = best_knn.score(X_test, y_test)

            # Log the performance metric
            accuracy = metrics.accuracy_score(y_test, results)
            precision = metrics.precision_score(y_test, results)
            recall = metrics.recall_score(y_test, results)
            f1 = metrics.f1_score(y_test, results)

            log_metric("accuracy", accuracy)
            log_metric("precision", precision)
            log_metric("recall", recall)
            log_metric("f1", f1)


            print(f"Accuracy: {accuracy}")
            print(f"Precision: {precision}")
            print(f"Recall: {recall}")
            print(f"F1 Score: {f1}")
            # Log the best model in MLflow
            mlflow.sklearn.log_model(best_knn, "best_knn")

            # Save the model to the MLflow artifact store
            mlflow.sklearn.save_model(best_knn, "best_knn_model")

            mlflow.end_run()



