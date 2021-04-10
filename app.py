# Importing all neccessary libraries
import streamlit as st 
import pandas as pd
import altair as alt
import docx2txt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, precision_score, recall_score,f1_score,roc_curve ,auc
from sklearn.ensemble import GradientBoostingClassifier
from imblearn.over_sampling import SMOTE
from sklearn import metrics
from PIL import Image
import numpy as np
import pickle

@st.cache # cashing dataframe
def loadData():
    df = pd.read_csv("files/dataframe.csv")
    return  df

@st.cache  
def preprocessing(df):# Preprocessing for ML model
    y = np.where(df['Attrition_Flag'] == 'Attrited Customer', 1, 0)
    # These are features selected using sequential feature selection from mlxtend library
    X = df[['Customer_Age','Total_Relationship_Count','Contacts_Count_12_mon', 'Credit_Limit', 'Total_Revolving_Bal', 'Avg_Open_To_Buy',
            'Total_Amt_Chng_Q4_Q1', 'Total_Trans_Amt', 'Total_Trans_Ct', 'Total_Ct_Chng_Q4_Q1', 'Income_Category']]
    X = pd.get_dummies(X)
    # Spliting X,y into training data and testing data
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=42)
    smote = SMOTE(sampling_strategy="minority")
    X_smt,y_smt = smote.fit_resample(X_train,y_train)
    return X_smt,X_test,y_smt,y_test
   
    
@st.cache(suppress_st_warning=True)
def Gradientboosting(X_smt, X_test, y_smt, y_test):
    # Loading already trained model with parameters (n_estimators=220, learning_rate= 0.3, max_depth=8,random_state=10)
    with open('files/gboot_model_pickle','rb') as f:
        boost = pickle.load(f)
    
    #calculating classfication metrics based on already trained model boost
    y_pred = boost.predict(X_test)
    score = metrics.accuracy_score(y_test, y_pred) 
    precision = precision_score(y_test,y_pred)
    recall = recall_score(y_test,y_pred)
    F1_score = f1_score(y_test,y_pred)
    
    return score, precision,recall,F1_score, boost

# Accepting user data for predicting its Member Type
def accept_user_data():
    left_side,right_side = st.beta_columns(2) #To make side by side input clumns

    Customer_Age = left_side.text_input("Customer_Age")
    Total_Relationship_Count = left_side.text_input("Total_Relationship_Count /Total nonumber of products held by the customer: ")
    Contacts_Count_12_mon = left_side.text_input("Contacts_Count_12_mon / Number of Contacts in the last 12 months ")
    Credit_Limit = left_side.text_input("Credit_Limit /Credit limit")
    Total_Revolving_Bal = left_side.text_input("Total_Revolving_Bal/Total revolving balance on the credit card ")
    Avg_Open_To_Buy = left_side.text_input("Avg_Open_To_Buy / Open to Buy Credit Line (Average of last 12 months")
    Total_Amt_Chng_Q4_Q1 = right_side.text_input("Total_Amt_Chng_Q4_Q1/Change in Transaction Amount (Q4 over Q1) ")
    Total_Trans_Amt = right_side.text_input("Total_Trans_Amt / Total Transaction amount (Last 12 months) ")
    Total_Trans_Ct = right_side.text_input("Total_Trans_Ct/Total Transaction Count (Last 12 months) ")
    Total_Ct_Chng_Q4_Q1 = right_side.text_input("Total_Ct_Chng_Q4_Q1/Change in Transaction Count (Q4 over Q1) ")
    Income_Category = right_side.selectbox("Income_category /Annual Income Category",options=['$120K +','$40K - $60K','$60K - $80K', '$80K - $120K','Less than $40K', 'Unknown'])
      
    # creating dummy variable for user input data(income category)
    if Income_Category == '$120K +':
        I_value = [1,0,0,0,0,0]
    elif Income_Category == '$40K - $60K':
        I_value = [0,1,0,0,0,0]
    elif Income_Category == '$60K - $80K':
        I_value = [0,0,1,0,0,0]
    elif Income_Category == '$80K - $120K':
        I_value = [0,0,0,1,0,0]
    elif Income_Category == 'Less than $40K':
        I_value = [0,0,0,0,1,0]
    else:
        I_value = [0,0,0,0,0,1]
       
    user_prediction_data = np.array([Customer_Age,Total_Relationship_Count,Contacts_Count_12_mon,Credit_Limit,Total_Revolving_Bal,Avg_Open_To_Buy,
                                     Total_Amt_Chng_Q4_Q1,Total_Trans_Amt,Total_Trans_Ct,Total_Ct_Chng_Q4_Q1,*I_value]).reshape(1,-1)
    

    return user_prediction_data


def main():
    st.markdown(""" <div style="background-color:#d3f4e7;padding:10px ">
                <h1 style=" color:#ff8043;text-align:center"> Prediction of Bank Customer Churn</h1> </div> """,unsafe_allow_html=True)
    
    data = loadData()
    X_smt,X_test,y_smt,y_test = preprocessing(data)
    score, precision,recall,F1_score, boost = Gradientboosting(X_smt, X_test, y_smt, y_test)
    #Make different page/sections for the app
    Menu = st.sidebar.radio("Menu",("About","Data and Methods","Visualization","Result and Conclusion","ML_GradientBoosting"))



    if (Menu == "About"):
        st.subheader("**Summary**")
        summary_text =docx2txt.process("files/summary.docx") # uploading text document
        st.markdown(summary_text)
             
        #Load image 
        img = Image.open("files/churn.png")
        st.image(img, width = 705)
        st.subheader("**Introduction**")
        st.write("""Customer churn refers to when a customer or subscriber terminate relationship with a company during a given time period. It is a major problem
            and one of the most important concerns for many companies. The ability to identify customers at high risk of churning as early as possible enables the 
            business to engage these customers and it is a great opportunity to increase customer loyalty and avoid loss of revenue.Having a robust and accurate
            churn prediction model helps businesses to take actions on a timely manner in order to prevent customers from leaving the company.""")


        
        st.write("""The main objective of this project is to develop a churn prediction model which assists Banks to predict customers who are 
        most likely to churn.The focus is to select classification algorithm with the best predictive power and to design an application 
        which can take input from user for prediction.  """)
    elif (Menu == "Data and Methods"):
        st.subheader("**Data**")
        st.markdown('''The data for this project is obtained from  https://www.kaggle.com/datasets. The dataset consists of 19 features about customers/members  of a bank with a binary target variable indicating
                    wheather the customer terminated membership or still active member. First 5 observations are shown here.''')
        
        st.write(data.head())# displaying the first 5 rows of data
        
        st.subheader("**Feature Selection**")
        raw_text =docx2txt.process("files/method.docx") # uploading text document
        
        st.markdown(f'<div style = text-align:justfy>{raw_text}</div>',unsafe_allow_html=True) # Display uploaded text data with justfy text-alignment 
        st.success("**Classfication Algorithms and cost of false prediction**  ")
        algorithms_text =docx2txt.process("files/algorithms_cost.docx")
        st.write(algorithms_text,unsafe_allow_html=True)

    # Fitting gradient boosting classfier 
    elif (Menu == "ML_GradientBoosting"):
        CHOOSE = st.sidebar.radio("CHOOSE", ('Gradient boosting', 'Predict churn with your input'))
                
        if (CHOOSE == "Gradient boosting"):
            st.subheader("**ROC curve for 4 classfiers**")
            st.text("Gradient boosting classfier has high auc, choosen as the best classfier.")
            img = Image.open("files/roc.png")
            st.image(img, width =400)

            #Display classfication metrics            
            st.markdown("**Summary of classification metrics for  Gradient boosting classifier:** ")
            st.write("**_Accuracy:_**", score)
            st.write("**_Precision:_**", precision)
            st.write("**_Recall:_**" ,recall)
            st.write("**_F1_score:_**", F1_score)
        elif (CHOOSE == "Predict churn with your input"):
            st.write('**Make sure for valid data input. Click checkbox to see 5 rows of the dataframe**')
            
            if st.checkbox("Show Data"):# Choice to display dataframe head
                st.text("First five rows of the data to help what your input should look like")
                st.write(data.head()) # display the first five rows from dataframe
                            
            try:
            
                user_prediction_data = accept_user_data()
                submit = st.button('PREDICT')
                if submit:
                    score, precision,recall,F1_score, boost = Gradientboosting(X_smt, X_test, y_smt, y_test)
                    pred = (boost.predict_proba(user_prediction_data)[0][1]).round(decimals=5, out=None)
                    st.write(" **predicted probability of churning for a customer with your input information is {}: **".format(pred))


                    st.text(" ------------------------------------------------------------------------------------------------")
                    st.write("compare the result with predicted probability distribution of test data for churned customers") 
                    st.text("25th quantile  =  0.815 , i.e 25% of churned customers predicted probability is less than 0.815")
                    st.text("50th quantile  =  0.91 i.e 50% of churned customers predicted probability is less than 0.91")
                    st.text("75th quantile  =  0.943 ,i.e 75% of churned customers predicted probability is less than 0.943")

                    
            except Exception as e: print(e)
        

    

    elif (Menu == "Visualization"):
        st.subheader("**Data Visualization**")
        Select_dropdown = st.selectbox('Select from dropdown',['Visual1','Visual2'])
        
        df2 = loadData()

        if Select_dropdown == 'Visual1':
            scale2 = alt.Scale(domain=['Existing Customer', 'Attrited Customer'],
                  range=[ '#1c30b7','#cbe111'])
            #Making interactive selection using scatter plot
            scatter_selector = alt.selection(type='interval', encodings=['x', 'y'], empty='all')
            base = alt.Chart(df2).mark_circle().encode(
                alt.X('Total_Trans_Amt', title='Total transaction amount'),
                alt.Y('Credit_Limit', title='Credit limit'),
                alt.Color('Attrition_Flag', scale=scale2)
            ).properties(
                height=200,
                width=300
            )
            columns = [0,1,2,3,4,5,6]
            # A dropdown filter
            column_dropdown = alt.binding_select(options=columns)
            column_select = alt.selection_single(
                fields=['Months_Inactive_12_mon'],
                on='doubleclick',
                clear=False,
                bind=column_dropdown,
                name='Number_of',
                init={'Months_Inactive_12_mon': 3}
            )
            # Specify the top chart as a modification of the base chart
            filter_columns = base.add_selection(
                column_select
            ).transform_filter(
                column_select
            ).properties(
                height=200,
                width=300
            ).add_selection(scatter_selector)
            # Specify the lower chart as a modification of the base chart
            lower = alt.Chart(df2, width=300, height=150).mark_bar().encode(
                alt.X("Income_Category"),
                alt.Y("count()"),
                alt.Color("Attrition_Flag:N",scale = scale2)
            ).add_selection(
                column_select
            ).transform_filter(
                column_select|scatter_selector
            )
            # Second side chart
            third = alt.Chart(df2, width=250,height=200).mark_bar().encode(
                alt.X("Customer_Age:Q",  bin=alt.Bin(maxbins=70)),
                alt.Y('count()', stack=None),
                alt.Color('Attrition_Flag:N',scale=scale2)
        
            ).add_selection(
                column_select
            ).transform_filter(
                column_select|scatter_selector
            )
            # Lower side chart

            fourth = alt.Chart(df2,width=250,height=150).mark_tick().encode(
                alt.X('Avg_Utilization_Ratio:Q',axis=alt.Axis(grid=False)),
                alt.Y('Card_Category:O',axis=alt.Axis(grid=False)),
                alt.Color('Attrition_Flag:N',scale = scale2)
            ).add_selection(
                column_select
            ).transform_filter(
                column_select|scatter_selector
            )
            filter_columns & lower |third & fourth


            st.text('Select different values from dropdown to see how visualization changes.')
                

        elif Select_dropdown == 'Visual2':
            scale2 = alt.Scale(domain=['Existing Customer', 'Attrited Customer'],
                  range=[ '#6293ff', '#ff8043'])
            base = alt.Chart(df2,width=300, height=200).mark_circle().encode(
                alt.X('Total_Trans_Amt', title=' Transactions amount in 12 months',axis=alt.Axis(grid=False)),
                alt.Y('Customer_Age', title='Customer Age',axis=alt.Axis(grid=False)),
                alt.Color('Attrition_Flag', scale=scale2),
            
            )
            chart2 =alt.Chart(df2,width=300, height=200).mark_circle().encode(
                alt.X('Months_on_book', title='Period with bank(months)',axis=alt.Axis(grid=False)),
                alt.Y('Total_Revolving_Bal',axis=alt.Axis(grid=False)),
                alt.Color('Attrition_Flag', scale=scale2)
                
            )
            bar = alt.Chart(df2,width=250, height=200).mark_bar().encode(
                            alt.X('Education_Level:N'),
                            alt.Y('count()'),
                            alt.Color ("Attrition_Flag")
                        ) 
            chart = alt.Chart(df2,width=250, height=200).mark_bar().encode(
                            alt.X('Total_Amt_Chng_Q4_Q1:Q',bin=alt.Bin(maxbins=70)),
                            alt.Y('count()'),
                            alt.Color ("Attrition_Flag")
            )                      

            base&bar|chart&chart2 

    
    elif (Menu == "Result and Conclusion"):
        st.subheader("**Results**")
        st.markdown("""From a total of 19 features top 11 features contributing the most in predicting churn were selected using wrapper method.
         Age, Credit limit,Total transaction amount, Income category, Total revolving balance are among the selected features.""",unsafe_allow_html=True)

        st.text("Table : Summary of classfication metrics for various classfiers")
        #Displaying table of different ML classfication metrics 
        img = Image.open("files/Classfication_metrics.png")
        st.image(img, width =650)
        #Displaying text summary of the result
        raw_text2 =docx2txt.process("files/result.docx")
        st.markdown(raw_text2)
        
        #Displaying confusion matrix picture
        img3 = Image.open("files/confusion_matrix.png")
        st.image(img3, width =650)
        st.write("""Using the model shows much saving when compared to total spending without using the model. 
                    Therefore, it is recommended to invest in optimizing the model further to decrease the cost.""")
        #Displaying precision recall curve picture
        img2 = Image.open("files/precision_recall.png")
        st.image(img2, width =650)
        st.text("Precision and Recall vs Threshold curve, and Precision-Recall curve for Graradient Boosting classifier ")




if __name__=='__main__':
    main()

