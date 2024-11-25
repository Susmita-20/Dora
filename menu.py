import streamlit as st
import streamlit.components.v1 as components



def authmenu():
    st.sidebar.page_link("./pages/project.py",label="Projects")
    st.sidebar.page_link("./pages/query.py",label="Query")
    st.sidebar.page_link("./pages/visualize.py",label="Data Analysis")
    
    # st.switch_page("./pages/project.py")

def unauthmenu():
    st.sidebar.page_link("./pages/authenticate.py",label="Login/Signup")
    # st.switch_page("./pages/authenticate.py")
    


def menu():
    HtmlFile = open('test.html', 'r', encoding='utf-8')
    source_code = HtmlFile.read()
    print(source_code)
    components.html(source_code, height=600)

    if "role" not in st.session_state or st.session_state.role is None:
        unauthmenu()
    else:
        authmenu()
        logout = st.sidebar.button("Logout")
        if logout:
            st.session_state.role = None
            st.session_state.projects = []
            st.session_state.curr = None
            st.session_state.messages = []
            st.session_state.clear()
            st.switch_page("./pages/authenticate.py")
    
        
    