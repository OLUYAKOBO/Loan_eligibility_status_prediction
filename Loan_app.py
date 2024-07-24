import pandas as pd
import pickle 
import streamlit as st
import numpy as np


scaler = pickle.load(open('scal.pkl','rb'))
model = pickle.load(open('loan_eli.pkl','rb'))
encoder = pickle.load(open('encoder.pkl','rb'))

st.title(" Loan Eligibility Status Application")

st.write("This App predicts the **Loan Eligibility** status of a customer")

st.header("User Information")

def user_info():
    
    c1,c2 = st.columns(2)
    
    with c1:
        gender = st.selectbox('Please select your gender', (['Male','Female']))
        mar = st.selectbox('Are you married?',(['Yes','No']))
        dep = st.selectbox('How many dependents do you have?', (['0','1','2','3+']))
        edu = st.selectbox('Are you a college graduate', (['Graduate','Not Graduate']))
        self_emp = st.selectbox('Are you self employed?', (['Yes','No']))
        
    with c2:
        app_inc = st.number_input('Please enter your annual income')
        coapp_inc = st.number_input('Please enter your coapplicant annual income')
        loan_amt = st.number_input('Please enter the amount of loan you want to obtain')
        term = st.number_input('Please enter your term',10,480,36)
        cred_hist = st.selectbox('Do you have a credit history', (['Yes','No']))
        
    area = st.selectbox('Which type of area do you live in',(['Urban','Semiurban','Rural']))
    
    feat = np.array([gender,
                     mar,
                     dep,
                     edu,
                     self_emp,
                     app_inc,
                     coapp_inc,
                     loan_amt,
                     term,
                     cred_hist,
                     area]).reshape(1,-1)
    
    cols = ['Gender',
            'Married',
            'Dependents',
            'Education',
            'Self_Employed',
            'Applicant_Income',
            'Coapplicant_Income',
            'Loan_Amount',
            'Term',
            'Credit_History',
            'Area'
           ]
    
    df = pd.DataFrame(feat, columns = cols)
    return df
df = user_info()

#st.write(df)
df.replace({'Yes':1,
            'No':0},
          inplace = True)

#st.write(df)

def encoding():
    df1 = df.copy()
    
    cat_cols = ['Gender','Dependents','Education','Area']
    encoded_data = encoder.transform(df1[cat_cols])
    dense_data = encoded_data.todense()
    df1_encoded = pd.DataFrame(dense_data, columns = encoder.get_feature_names_out())
    
    df1 = pd.concat([df1,df1_encoded],
                    axis = 1)
    df1.drop(cat_cols,
             axis = 1,
             inplace = True)
    
    cols = df1.columns
    df1 = scaler.transform(df1)
    df1 = pd.DataFrame(df1,columns=cols)
    
    return df1
df1 = encoding()

model = pickle.load(open('loan_eli.pkl','rb'))
prediction = model.predict(df1)
           
import time

if st.button('*Click here to know your Loan Eligibility Status*'):
    time.sleep(10)
    with st.spinner('Predicting... Please wait...'):
        if prediction == 0:
            st.success("You are not eligible for a loan")
        else:
            st.success("You are eligible for a loan")
    
               
            
            
            
            
            
            
