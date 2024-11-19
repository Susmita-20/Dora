import streamlit as st
import os
import pyrebase
from menu import menu

if "role" not in st.session_state:
    st.session_state.role = None


st.set_page_config(page_title="DORA", page_icon="🦙")
st.markdown(f"""<style>
        .st-emotion-cache-79elbk{{
            display: none;}}
            </style>""", unsafe_allow_html=True)
menu()

if "projects" not in st.session_state:
        st.session_state.projects = []

firebaseConfig = {
  'apiKey': st.secrets["apiKey"],
  'authDomain': st.secrets["authDomain"],
  'projectId': st.secrets["projectId"],
  'storageBucket': st.secrets["storageBucket"],
  'messagingSenderId': st.secrets["messagingSenderId"],
  'appId': st.secrets["appId"],
  'measurementId': st.secrets["measurementId"],
  'databaseURL': st.secrets["databaseURL"]
  }

firebase = pyrebase.initialize_app(firebaseConfig)
auth = firebase.auth()

def signup(email,password):
    user = auth.create_user_with_email_and_password(email, password)
    return user

def login(email,password):
    try:
        user = auth.sign_in_with_email_and_password(email, password)
        st.success("Successfully Logged In")
        return user
    except:
        return "Invalid Credentials"

type = st.selectbox("Sign Up/Log In", ["Sign Up", "Log In"])
if type == "Log In":
    st.header("Login")
    email = st.text_input("Email")
    password = st.text_input("Password", type='password')
    login_button = st.button("Login")
    if login_button:
        with st.spinner("Taking you to the Dashboard..."):
            user = login(email,password)
        if user == "Invalid Credentials":
            st.error("Invalid Credentials")
        else:
            st.success("Successfully Logged In")
            st.session_state.role = user['email']
            if os.path.exists(f"{st.session_state.role}"):
                projects = os.listdir(f"{st.session_state.role}")
                for project in projects:
                    if project != 'index' and project != 'summary':
                        st.session_state.projects.append(project)
            st.switch_page("./pages/project.py")
if type == "Sign Up":
    st.header("Sign Up")
    email = st.text_input("Email")
    password = st.text_input("Password", type='password')
    signup_button = st.button("Sign Up")
    if signup_button:
        with st.spinner("Onboarding you..."):
            user = signup(email,password)
        if user == "User already exists":
            st.error("User already exists")
        else:
            st.success("Successfully Signed Up")
            st.session_state.role = user['email']
            st.switch_page("./pages/project.py")







    


