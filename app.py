import streamlit as st
import pandas as pd
from langchain_community.document_loaders import CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
import streamlit as st
groq_api_key = st.secrets["GROQ_API_KEY"]
st.set_page_config(page_title="FinSight AI")
st.title("FinSight AI - Startup Funding Assistant")



@st.cache_resource
def load_db():
    loader = CSVLoader(file_path="startup_data.csv")
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = Chroma.from_documents(chunks, embeddings)
    return db

db = load_db()
retriever = db.as_retriever(search_kwargs={"k": 3})

query = st.text_input("Ask (e.g. funding for edtech?)")

if query:
    docs = retriever.invoke(query)
    context = "\n\n".join([d.page_content for d in docs])
    
    llm = ChatGroq(groq_api_key=GROQ_KEY, model_name="openai/gpt-oss-20b")
    
    prompt = f"Context: {context}\n\nQuestion: {query}\n\nAnswer using world knowledge+context.Give global startups list:"
    response = llm.invoke(prompt)
    
    st.write(response.content)
    with st.expander("Source Data"):
        st.write(context)