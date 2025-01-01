import streamlit as st
from home import display_home
from sentiment_analysis import display_sentiment_analysis
from about import display_about

# Set up page configuration
st.set_page_config(
    page_title="Ai finance assistant",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="auto",
    menu_items={'About': "Gopinath creation"})

# Set title 
st.markdown('''<h1 style='font-family: "Arial"; color: #1fcae5;'>AI Finance assistant 📈 </h1>''',unsafe_allow_html=True)

# Set up columns for navigation
col1, col2, col3 = st.columns(3)

# Navigation buttons
with col1:
    if st.button("**Home**",use_container_width=True):
        st.session_state.page = "Home"

with col2:
    if st.button("**News Sentiment Analyzer**",use_container_width=True):
        st.session_state.page = "News Sentiment Analyzer"

with col3:
    if st.button("**About**",use_container_width=True):
        st.session_state.page = "About"

# Default to Home page
if "page" not in st.session_state:
    st.session_state.page = "Home"

# Render content based on selected page
if st.session_state.page == "Home":
    display_home()
elif st.session_state.page == "News Sentiment Analyzer":
    display_sentiment_analysis()
elif st.session_state.page == "About":
    display_about()