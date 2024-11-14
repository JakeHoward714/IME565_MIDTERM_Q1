# Import necessary libraries
import streamlit as st
import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Suppress warnings
import warnings
warnings.filterwarnings("ignore")

# Set up the title and description of the app
st.title('Fetal Health Classification: A Machine Learning App') 

#Display gif
st.image('fetal_health_image.gif', use_column_width=True)

st.write("Utilize our advance Machine Learning application to predict fetal health classifications.")

# Create a sidebar for input collection
st.sidebar.header('Fetal Health Features Input')

# Create csv upload for data
health_upload = st.sidebar.file_uploader("Upload your data", help='File must be in CSV format')

 #DISPLAY EXAMPLE CSV
st.sidebar.warning('Esnure your data strictly follows the format outlined below')
# Import example dataframe
df = pd.read_csv('fetal_health.csv')

 # Format correctly
df_clean = df.drop(columns=['fetal_health'])
#Write to streamlit
st.sidebar.write(df_clean.head())

# Create Selection Box for Model Type
st.sidebar.write("**Model Selection**")
model_type = st.sidebar.radio('Choose model for prediciton', options = ['Decision Tree', 'Random Forest','AdaBoost','Soft Voting'])
st.sidebar.info(f'You Selected: {model_type}')


# RUN PREDICTIONS
# Check to see if csv file as been uploaded
if health_upload is None:
   #Tell to upload data
   st.info('Please upload data to proceed')
else:
   # Say success
   st.success('CSV file uploaded successfully')
   user_df = pd.read_csv(health_upload)

   # Generate prediction based on model selection
   if model_type == 'Decision Tree':
        #Open decision tree model
        dt_pickle = open('decision_tree_health.pickle', 'rb') 
        clf = pickle.load(dt_pickle) 
        dt_pickle.close()

   elif model_type == 'Random Forest':
        #Open decision tree model
        dt_pickle = open('random_forest_health.pickle', 'rb') 
        clf = pickle.load(dt_pickle) 
        dt_pickle.close()

   elif model_type == 'AdaBoost':
       #Open decision tree model
        dt_pickle = open('AdaBoost_health.pickle', 'rb') 
        clf = pickle.load(dt_pickle) 
        dt_pickle.close()
   elif model_type == 'Soft Voting':
       #Open decision tree model
        dt_pickle = open('SoftVote_health.pickle', 'rb') 
        clf = pickle.load(dt_pickle) 
        dt_pickle.close()

   #Run predictions for each row of data
   # Use predict() on the entire DataFrame at once
   predictions = clf.predict(user_df)

   # Get probability scores for each class
   prob_scores = clf.predict_proba(user_df)

   #Write predictions and scores back to user data frame
   user_df['Predicted Fetal Health'] = predictions
   user_df['Prediction Probability (%)'] = np.max(prob_scores, axis = 1) * 100

   # Color code the predicted output accordingly
   # ***UTILIZED CHAT GPT FOR HELP WITH SYNTAX FOR COLOR CODING THE OUTPUT***

   # Define a function for applying the background color based on the Predicted Class value
   def color_predicted_class(val):
     color = ''
     if val == 'Normal':
          color = 'lime'
     elif val == 'Suspect':
          color = 'yellow'
     elif val == 'Pathological':
          color = 'orange'
     return f'background-color: {color}'

    # Apply the background color to dataframe
   df_dislay = user_df.style.applymap(color_predicted_class, subset=['Predicted Fetal Health'])

   #Display results to streamlit
   st.header(f'Predicting Fetal Health Class Using {model_type} Model')
   st.write(df_dislay)

   # Showing additional items in tabs
   st.subheader("Model Performance and Insights")
   tab1, tab2, tab3 = st.tabs(["Confusion Matrix", "Classification Report", "Feature Importance"])
    
   if model_type == 'Decision Tree':
        # Tab 1: Confusion Matrix
        with tab1:
            st.write("### Confusion Matrix")
            st.image('confusion_mat_dt.svg')
            st.caption("Confusion Matrix of model predictions.")

        #Tab 2: Classification Report
        with tab2:
            st.write("### Classification Report")
            report_df = pd.read_csv('class_report_health_tree.csv', index_col = 0).transpose()
            st.dataframe(report_df.style.background_gradient(cmap='RdBu').format(precision=2))
            st.caption("Classification Report: Precision, Recall, F1-Score, and Support for each Health Classification.")
          
        # Tab 3: Feature Importance 
        with tab3:
            st.write("### Feature Importance")
            st.image('feature_imp_health_tree.svg')
            st.caption("Features used in this prediction are ranked by relative importance.")

   elif model_type == 'Random Forest':
        # Tab 1: Confusion Matrix
        with tab1:
            st.write("### Confusion Matrix")
            st.image('confusion_mat_rf.svg')
            st.caption("Confusion Matrix of model predictions.")

        #Tab 2: Classification Report
        with tab2:
            st.write("### Classification Report")
            report_df = pd.read_csv('class_report_health_rf.csv', index_col = 0).transpose()
            st.dataframe(report_df.style.background_gradient(cmap='RdBu').format(precision=2))
            st.caption("Classification Report: Precision, Recall, F1-Score, and Support for each Health Classification.")
          
        # Tab 3: Feature Importance 
        with tab3:
            st.write("### Feature Importance")
            st.image('feature_imp_health_rf.svg')
            st.caption("Features used in this prediction are ranked by relative importance.")

   elif model_type == 'AdaBoost':
        # Tab 1: Confusion Matrix
        with tab1:
            st.write("### Confusion Matrix")
            st.image('confusion_mat_ada.svg')
            st.caption("Confusion Matrix of model predictions.")

        #Tab 2: Classification Report
        with tab2:
            st.write("### Classification Report")
            report_df = pd.read_csv('class_report_health_ada.csv', index_col = 0).transpose()
            st.dataframe(report_df.style.background_gradient(cmap='RdBu').format(precision=2))
            st.caption("Classification Report: Precision, Recall, F1-Score, and Support for each Health Classification.")
          
        # Tab 3: Feature Importance 
        with tab3:
            st.write("### Feature Importance")
            st.image('feature_imp_health_ada.svg')
            st.caption("Features used in this prediction are ranked by relative importance.")

   elif model_type == 'Soft Voting':
        # Tab 1: Confusion Matrix
        with tab1:
            st.write("### Confusion Matrix")
            st.image('confusion_mat_soft.svg')
            st.caption("Confusion Matrix of model predictions.")

        #Tab 2: Classification Report
        with tab2:
            st.write("### Classification Report")
            report_df = pd.read_csv('class_report_health_vote.csv', index_col = 0).transpose()
            st.dataframe(report_df.style.background_gradient(cmap='RdBu').format(precision=2))
            st.caption("Classification Report: Precision, Recall, F1-Score, and Support for each Health Classification.")
          
        # Tab 3: Feature Importance 
        with tab3:
            st.write("### Feature Importance")
            st.image('feature_imp_health_vote.svg')
            st.caption("Features used in this prediction are ranked by relative importance.")


       
       

