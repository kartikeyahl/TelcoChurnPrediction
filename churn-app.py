import streamlit as st
import sqlite3
from passlib.hash import pbkdf2_sha256
import pandas as pd
import numpy as np
import pickle
import base64
import seaborn as sns
import matplotlib.pyplot as plt

def create_users_table():
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  username TEXT UNIQUE NOT NULL,
                  password_hash TEXT NOT NULL)''')
    conn.commit()
    conn.close()

def add_user(username, password):
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    password_hash = pbkdf2_sha256.hash(password)
    c.execute('INSERT INTO users (username, password_hash) VALUES (?, ?)', (username, password_hash))
    conn.commit()
    conn.close()

def verify_password(username, password):
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute('SELECT password_hash FROM users WHERE username=?', (username,))
    row = c.fetchone()
    conn.close()
    if row:
        return pbkdf2_sha256.verify(password, row[0])
    else:
        return False

def login():
    st.subheader("Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type='password')
    if st.button("Login"):
        if verify_password(username, password):
            st.session_state['username'] = username
            st.success("Logged in as {}".format(username))
            st.experimental_rerun()
        else:
            st.error("Incorrect username or password")

def signup():
    st.subheader("Create New Account")
    username = st.text_input("Username")
    password = st.text_input("Password", type='password')
    confirm_password = st.text_input("Confirm Password", type='password')
    if st.button("Sign up"):
        if password == confirm_password:
            try:
                add_user(username, password)
                st.success("Account created for {}".format(username))
            except sqlite3.IntegrityError:
                st.error("Username already taken, please choose another")
        else:
            st.error("Passwords do not match")

def user_home():
    st.subheader("Hello, {}".format(st.session_state['username']))
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)

    st.write("""
    # Churn Prediction App

    Customer churn is defined as the loss of customers after a certain period of time. Companies are interested in targeting customers
    who are likely to churn. They can target these customers with special deals and promotions to influence them to stay with
    the company. 

    This app predicts the probability of a customer churning using Telco Customer data. Here
    customer churn means the customer does not make another purchase after a period of time. 

    """)



    df_selected = pd.read_csv("telco_churn.csv")
    df_selected_all = df_selected[['gender', 'Partner', 'Dependents', 'PhoneService', 
                                        'tenure', 'MonthlyCharges', 'Churn']].copy()


    st.set_option('deprecation.showPyplotGlobalUse', False)
   




    def user_input_features():
        gender = st.sidebar.selectbox('gender',('Male','Female'))
        PaymentMethod = st.sidebar.selectbox('PaymentMethod',('Bank transfer (automatic)', 'Credit card (automatic)', 'Mailed check', 'Electronic check'))
        MonthlyCharges = st.sidebar.slider('Monthly Charges', 18.0,118.0, 18.0)
        tenure = st.sidebar.slider('tenure', 0.0,72.0, 0.0)

        data = {'gender':[gender], 
                'PaymentMethod':[PaymentMethod], 
                'MonthlyCharges':[MonthlyCharges], 
                'tenure':[tenure],}
        
        features = pd.DataFrame(data)
        return features
    input_df = user_input_features()



    churn_raw = pd.read_csv('telco_churn.csv')




    churn_raw.fillna(0, inplace=True)
    churn = churn_raw.drop(columns=['Churn'])
    df = pd.concat([input_df,churn],axis=0)




    encode = ['gender','PaymentMethod']
    for col in encode:
        dummy = pd.get_dummies(df[col], prefix=col)
        df = pd.concat([df,dummy], axis=1)
        del df[col]
    df = df[:1] # Selects only the first row (the user input data)
    df.fillna(0, inplace=True)


    features = ['MonthlyCharges', 'tenure', 'gender_Female', 'gender_Male',
        'PaymentMethod_Bank transfer (automatic)',
        'PaymentMethod_Credit card (automatic)',
        'PaymentMethod_Electronic check', 'PaymentMethod_Mailed check']

    df = df[features]



    # Displays the user input features
    st.subheader('User Input features')
    print(df.columns)

    st.write(df)

    # Reads in saved classification model
    load_clf = pickle.load(open('churn_clf.pkl', 'rb'))

    # Apply model to make predictions
    prediction = load_clf.predict(df)
    prediction_proba = load_clf.predict_proba(df)


    st.subheader('Prediction')
    churn_labels = np.array(['No','Yes'])
    st.write(churn_labels[prediction])

    st.subheader('Prediction Probability')
    st.write(prediction_proba)



    if st.button("Log out"):
        del st.session_state['username']
        st.experimental_rerun()  # reset the app

create_users_table()
st.set_page_config(page_title="Login/Signup App", page_icon=":guardsman:", layout="wide")
if 'username' not in st.session_state:
    option = st.sidebar.selectbox("Select an option", ["Login", "Create Account"])
    if option == "Login":
        login()
    else:
        signup()
else:
    user_home()
