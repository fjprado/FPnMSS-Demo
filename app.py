from dotenv import load_dotenv
from helpers import document_chunk, search_docs, summarize_text

load_dotenv()

import streamlit as st
import os
import pandas as pd
import io
import PyPDF2
from search_setup import load_search_index

st.set_page_config(page_title="FPnMSS - OpenAI Demo")
st.header("FPnMSS - OpenAI Demo")
st.caption("A simple project that gets data from a PDF file and uses it as a knowledge base to answer questions from the user.")
st.divider()
uploaded_file = st.file_uploader(label="Upload a PDF file", accept_multiple_files=False)
combined_documents = ''

if uploaded_file is not None:
    # To read file as bytes:
    bytes_data = uploaded_file.getvalue()
    pdf_file_like = io.BytesIO(bytes_data)
        
    # creating a pdf reader object
    pdfReader = PyPDF2.PdfReader(pdf_file_like)
    
    # creating a page object
    pageObj = pdfReader.pages[0]
    
    doc_text = pageObj.extract_text()   
    
    # get current data from search_index by file_name   
    # search_result = search_index(uploaded_file.name)
    
    # chunk documents and apply embeddings
    documents = document_chunk(doc_text, uploaded_file.name)   
    
    # add documents to search_index
    # load_search_index(documents) 
    
    # merge current documents with AI Search index results
    combined_documents = documents #+ search_result     

input = st.text_area("Question (Ask for something about your document):", key="input")

submit = st.button("Send :rocket:")

if submit:   
    # search for similar documents embedding user input
    res = search_docs(combined_documents, input, top_n=4)
    
    # summarize response and show to user
    summary = summarize_text(res, input)    
    
    st.subheader("the response is")
    st.write(summary)