import streamlit as st
import os
from dotenv import load_dotenv
import asyncio

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import scipy
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import requests
from bs4 import BeautifulSoup
import re
from PyPDF2 import PdfReader

# LangChain and other dependencies
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.chains import RetrievalQA


def display_sentiment_analysis():
    st.header(":red[News Sentiment Analyzer]")
    st.write("Analyze sentiment from financial news to make informed trading decisions.")

    # Sidebar for
    st.sidebar.header("Choose Model")
    model_selection = st.sidebar.radio("Select a loader", ("Pre trained Model (FinBERT)", "Gen Ai Model (Llama3.1:8b)"))

    if model_selection == "Pre trained Model (FinBERT)":
        # Load FinBERT Model
        @st.cache_resource
        def load_model():
            tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
            model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
            return tokenizer, model

        tokenizer, model = load_model()

        # Function for Sentiment Analysis
        def analyze_sentiment(text, tokenizer, model):
            tokenizer_kwargs = {"padding": True, "truncation": True, "max_length": 512}
            with torch.no_grad():
                input_sequence = tokenizer(text, return_tensors="pt", **tokenizer_kwargs)
                logits = model(**input_sequence).logits
                scores = {
                    k: v
                    for k, v in zip(
                        model.config.id2label.values(),
                        scipy.special.softmax(logits.numpy().squeeze()),
                    )
                }
            sentiment = max(scores, key=scores.get)
            probability = max(scores.values())
            return sentiment, probability

        # Word Cloud Plotting
        def plot_wordcloud(text):
            wordcloud = WordCloud(width=800, height=400, background_color="white").generate(text)
            plt.figure(figsize=(10, 5))
            plt.imshow(wordcloud, interpolation="bilinear")
            plt.axis("off")
            st.pyplot(plt)

        # Text Extraction from URL
        def extract_text_from_url(url):
            response = requests.get(url)
            soup = BeautifulSoup(response.content, "html.parser")
            text = soup.get_text()
            text = re.sub(r"\s+", " ", text)  # Clean extra spaces
            return text

        # Text Extraction from PDF
        def extract_text_from_pdf(uploaded_file):
            pdf_reader = PdfReader(uploaded_file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()
            return text

        
        st.sidebar.header("Choose Data Source")
        data_source = st.sidebar.radio("Select a loader", ("PDF", "URL"))

        text = None

        # PDF Loader
        if data_source == "PDF":
            uploaded_file = st.sidebar.file_uploader("Upload a PDF file", type="pdf")
            if uploaded_file:
                with st.spinner("Processing PDF..."):
                    try:
                        text = extract_text_from_pdf(uploaded_file)
                        st.success("PDF Processed Successfully!")
                    except Exception as e:
                        st.error(f"Error processing PDF: {e}")

        # URL Loader
        elif data_source == "URL":
            url = st.sidebar.text_input("Enter a URL to load content from")
            if st.sidebar.button("Load URL"):
                with st.spinner("Processing URL..."):
                    try:
                        text = extract_text_from_url(url)
                        st.success("URL Processed Successfully!")
                    except Exception as e:
                        st.error(f"Error processing URL: {e}")

        # Analyze and Display Results
        if text:
            st.subheader("Extracted Text")
            st.write(text[:500] + "...")  # Display first 500 characters

            st.subheader("Word Cloud")
            plot_wordcloud(text)

            st.subheader("Sentiment Analysis")
            sentiment, probability = analyze_sentiment(text, tokenizer, model)
            st.write(f"**Sentiment:** {sentiment}")
            st.write(f"**Probability:** {probability:.2f}")
            


    if model_selection == "Gen Ai Model (Llama3.1:8b)":

        # Sidebar for data source selection
        st.sidebar.header("Choose Data Source")
        data_source = st.sidebar.radio("Select a loader", ("PDF", "URL"))

        # Load environment variables
        try:
            load_dotenv()
            os.environ["LANGCHAIN_TRACING_V2"] = "true"
            os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
        except Exception as e:
            st.error(f"Langchain API key or .env file is unavailable at the specified file location.{e}")

        # Initialize session state variables
        if "llama_model" not in st.session_state:
            st.session_state["llama_model"] = "llama3.1:8b"
        if "messages" not in st.session_state:
            st.session_state.messages = []
        if "knowledge_base" not in st.session_state:
            st.session_state.knowledge_base = None

        # Load the language model
        llm = Ollama(model=st.session_state["llama_model"], temperature=0)
        output_parser = StrOutputParser()
        prompt_template = ChatPromptTemplate.from_messages(
            [
                ("system", "You are a helpful Finance assistant and News Sentiment Analyzer. Please respond to user queries."),
                ("user", "query: {question}")
            ]
        )
        chain = prompt_template | llm | output_parser

        # Function to set up the QA retrieval chain
        def setup_retrieval_qa_chain(knowledge_base):
            return RetrievalQA.from_chain_type(chain, retriever=knowledge_base.as_retriever())

        # Load data and create embeddings
        def process_text(text):
            text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
                encoding_name="cl100k_base", chunk_size=1000, chunk_overlap=250
            )
            texts = text_splitter.split_text(text)
            embeddings = HuggingFaceEmbeddings()
            documents = [Document(page_content=chunk) for chunk in texts]
            return FAISS.from_documents(documents, embeddings)

        # PDF Loader
        if data_source == "PDF":
            uploaded_file = st.sidebar.file_uploader("Upload a PDF file", type="pdf")
            if uploaded_file:
                with st.spinner("Processing PDF..."):
                    try:
                        # Save the uploaded file temporarily
                        with open(uploaded_file.name, "wb") as f:
                            f.write(uploaded_file.getbuffer())

                        # Load the PDF using PyPDFLoader
                        loader = PyPDFLoader(uploaded_file.name)
                        pages = []
                        
                        async def load_pdf_pages():
                            async for page in loader.alazy_load():
                                pages.append(page)
                        
                        asyncio.run(load_pdf_pages())

                        # Combine all page contents into a single string
                        combined_text = ''.join(page.page_content for page in pages)
                        st.session_state.knowledge_base = process_text(combined_text)
                        
                        st.success("PDF Uploaded Successfully! Data Extraction and Knowledge Base Creation Complete")
                        
                        # Optionally, delete the file after processing
                        os.remove(uploaded_file.name)
                    except Exception as e:
                        st.error(f"Error loading PDF: {e}")


        # URL Loader
        elif data_source == "URL":
            url = st.sidebar.text_input("Enter a URL to load content from")
            if st.sidebar.button("Load URL", use_container_width=True):
                with st.spinner("Processing URL..."):
                    try:
                        loader = WebBaseLoader(web_paths=[url])
                        docs = []
                        async def load_url_content():
                            async for doc in loader.alazy_load():
                                docs.append(doc)
                        asyncio.run(load_url_content())
                        
                        if docs:
                            doc = docs[0]
                            combined_text = (doc.metadata.get('source', '') +
                                            doc.metadata.get('title', '') +
                                            doc.metadata.get('description', '') +
                                            doc.page_content)
                            st.session_state.knowledge_base = process_text(combined_text)
                            st.success("URL Uploaded Successfully! Data Extraction and Knowledge Base Creation Complete")
                        else:
                            st.error("No content found at the specified URL.")
                    except Exception as e:
                        st.error(f"Error loading URL: {e}")

        # Check if knowledge base is loaded
        qa_chain = None
        if st.session_state.knowledge_base:
            qa_chain = setup_retrieval_qa_chain(st.session_state.knowledge_base)

        # Display chat history
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Chatbot interface
        if user_input := st.chat_input("What's on your mind?"):
            st.session_state.messages.append({"role": "user", "content": user_input})
            with st.chat_message("user"):
                st.markdown(user_input)

            # Get response from the RAG application
            if qa_chain:
                try:
                    response = qa_chain.invoke({"query": user_input})['result']
                    with st.chat_message("assistant"):
                        st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                except Exception as e:
                    st.error(f"Error in processing query: {e}")
            else:
                st.error("No knowledge base loaded. Please upload a PDF or load a URL.")