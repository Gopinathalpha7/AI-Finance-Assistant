import streamlit as st
from PIL import Image

def display_about():
    st.header(":red[About]")
    st.markdown('''The AI Finance Assistant is an intelligent tool designed to simplify and enhance the process of financial sentiment analysis. 
               Built with cutting-edge AI technologies, it provides users with actionable insights by analyzing the sentiment of financial news, 
               articles, and other market-relevant content.''')  

    st.subheader(":blue[Why Choose AI Finance Assistant?]")
    st.markdown('''
               - **Accurate Sentiment Analysis**: Identify whether financial content conveys a positive, negative, or neutral tone with precision.  
               - **Dual AI Models**:  
                 - **FinBERT**: A specialized NLP model for financial text sentiment analysis, ensuring domain-specific accuracy.  
                 - **Llama Model**: A generative AI approach for advanced sentiment understanding and nuanced insights.  
               - **User-Friendly Interface**: Access sentiment reports and analytics through an intuitive and easy-to-use platform.  
               - **Real-Time Analysis**: Get results quickly, enabling you to respond promptly to market trends.''')  

    st.subheader(":blue[Who Can Benefit?]")  
    st.markdown('''            
               - **Investors**: Gain deeper insights into market sentiment to make informed investment decisions.  
               - **Financial Analysts**: Save time and improve efficiency with automated sentiment analysis tools.  
               - **Market Enthusiasts**: Stay informed about the emotional pulse of the financial world.  

               The AI Finance Assistant is here to support you in navigating the ever-changing financial landscape with confidence. 
               Empower your decision-making process with reliable sentiment insights.''')

    # Open an image file using PIL
    About_background_image = Image.open("Image/about_page.jpg")
    # Display the image
    st.image(About_background_image, use_column_width=True)

    sentiment_image = Image.open("Image/Sentiment_Analysis.jpg")
    st.image(sentiment_image, use_column_width=True)

    st.header(":red[Specification Details]")

    st.subheader(":blue[AI Sentiment Analysis using Fin BIRD]")

    st.markdown('''
    **Model Name:** ProsusAI/FinBERT  
    **Model Size:** Built on BERT-base architecture  
    **Parameters:** 110 Million (from the original BERT model)  
    **Number of Layers:** 12 (Transformer-based architecture)  
    **Context Length:** 512 tokens  
    **Architecture Type:** Transformer-based NLP model  
    **Training Data:** Financial corpora, including the Financial PhraseBank and other domain-specific datasets  
    **Fine-tuning:** Optimized for financial sentiment analysis tasks  
    **Use Case:** Sentiment analysis for financial news, earnings reports, and market insights  
    **Language:** English (Specialized for financial terminology)  
    **Output:** Sentiment classification (Positive, Negative, Neutral) with probabilities for each label  
    **Deployment:** Compatible with cloud-based or local hosting environments  
    ''')



    st.subheader(":blue[AI Sentiment Analysis using RAG]")

    st.markdown('''
    **Model Name:** Meta Llama 3.1 8B Instruct  
    **Model Size:** 4.92 GB  
    **Parameters:** 8 Billion  
    **Number of Layers:** 36  
    **Context Length:** 4096 tokens  
    **Architecture Type:** Transformer-based LLM  
    **Training Data:** Financial and General Domain Texts  
    **Fine-tuning:** Instruction-tuned for financial sentiment analysis tasks  
    **Use Case:** Sentiment analysis of financial news, market insights, financial Q&A  
    **Language:** English (Finance Terminology Support)  
    **Deployment:** Locally hosted on your machine  
    ''')


    
    with st.sidebar:
        # Title for the section
        st.title("Contact Details")

        # Contact information
        st.write("**Name:** Gopinath, Data Scientist")
        st.write("**Email:** [gopinathaiml12@gmail.com](mailto:gopinathaiml12@gmail.com)")
        st.write("**LinkedIn:** [Gopinath .](https://www.linkedin.com/in/gopinathaiml12/)")
        st.write("**Portfolio:** [gopinathportfolio.com](https://www.gopinathportfolio.com)")  # Replace with actual portfolio link
        st.write("**GitHub:** [Gopinathalpha7](https://github.com/Gopinathalpha7)")
        # Add more contact details here
    # Add your About and Contact page content here