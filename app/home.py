import streamlit as st
from PIL import Image

def display_home():
    st.header(":red[Home]")
    st.subheader(":blue[Welcome to the AI Finance Assistant]")
    st.markdown('''Welcome to the AI Finance Assistant, your go-to solution for analyzing sentiment in financial news and articles. 
               This innovative tool is designed to empower investors, financial analysts, and market enthusiasts 
               by providing actionable insights based on sentiment analysis.  
               Our assistant leverages two cutting-edge approaches for sentiment analysis:  
               - **FinBERT**: Specially designed for financial text, it analyzes news and articles to determine sentiment as positive, negative, or neutral.  
               - **Llama Model**: A state-of-the-art generative AI tool that offers advanced sentiment analysis for deeper insights.  
               With a user-friendly interface and real-time analytics, the AI Finance Assistant is your trusted partner in navigating the complexities of the financial world. 
               Make smarter decisions with precise and reliable sentiment insights.''')

    # Home page Image
    Homepage_image = Image.open("Image/Home_page_image_1.jpg")
    st.image(Homepage_image, use_column_width=True)