import os
import re
import fitz
import tempfile
import time
import streamlit as st
from dotenv import load_dotenv
from langchain.schema import Document
from langchain.vectorstores.qdrant import Qdrant as QdrantVectorStore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from qdrant_client import QdrantClient
import google.generativeai as genai
from langchain_community.document_loaders import PyPDFLoader

# Load API key
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)

# Utility Functions
def remove_surrogates(text: str) -> str:
    return re.sub(r'[\ud800-\udfff]', '', text)

#Function to extract texts from pdf
def extract_text_from_pdf(file):
    doc = fitz.open(stream=file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text

# UI Header
st.title("Paper Query")
st.header("Your own AI-powered PDF Assistant")

file = st.file_uploader("📎 Upload your PDF", type="pdf")

# Init session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "pdf_ready" not in st.session_state:
    st.session_state.pdf_ready = False

# Process PDF
if file and st.button("📚 Prepare PDF"):
    with st.spinner("Please wait. You PDF is being processed..."):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(file.getbuffer())
            tmp_path = tmp.name

        # Load PDF
        loader = PyPDFLoader(tmp_path)
        pages = loader.load()

        # Split & clean
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_documents(pages)
        for chunk in chunks:
            chunk.page_content = remove_surrogates(chunk.page_content)

        # Delete temp file
        os.remove(tmp_path)

    # Embeddings
    embedding_model = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=GEMINI_API_KEY
    )
    qdrant_client = QdrantClient(
    url="https://3e3583cf-d0e6-4cc1-b344-1dec8fdce1cc.us-east4-0.gcp.cloud.qdrant.io:6333", 
    api_key=QDRANT_API_KEY,
    )
    collection_name = "rag-chat-application"

    if qdrant_client.collection_exists(collection_name):
        qdrant_client.delete_collection(collection_name)

    with st.spinner("Vectorizing and indexing your PDF..."):
        vector_store = QdrantVectorStore.from_documents(
            documents=chunks,
            url="https://3e3583cf-d0e6-4cc1-b344-1dec8fdce1cc.us-east4-0.gcp.cloud.qdrant.io:6333",
            api_key=QDRANT_API_KEY,
            collection_name=collection_name,
            embedding=embedding_model,
            force_recreate=True
        )

    st.success("✅Successfully Processed!")
    # Save to session
    st.session_state.pdf_ready = True
    st.session_state.embedding_model = embedding_model
    st.session_state.vector_db = vector_store
    st.session_state.model = genai.GenerativeModel(model_name="gemini-1.5-flash")
    st.session_state.chat = st.session_state.model.start_chat()

    st.success("✅ PDF is ready to chat!")

# Wait message
if file and not st.session_state.pdf_ready:
    st.info("👆 Please click 'Prepare PDF' to begin chatting.")

# Chat Logic
if st.session_state.pdf_ready:
    # Show chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # User input box
    if prompt := st.chat_input("Ask something about your PDF..."):
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Similarity search
        vector_db = st.session_state.vector_db
        search_results = vector_db.similarity_search(query=prompt, k=3)

        context = "\n\n\n".join([
            f"Contents: {doc.page_content[:1000]}...\nPage: {doc.metadata.get('page_label', 'N/A')}\nSource: {doc.metadata.get('source', 'N/A')}"
            for doc in search_results
        ])

        SYSTEM_PROMPT = f"""
        You are a helpful AI Assistant who answers user queries based ONLY on the context retrieved from a PDF file.
        Always mention the page number and keep your answers precise and helpful.

        Context:
        {context}
        """

        # Full prompt to Gemini
        full_message = f"{SYSTEM_PROMPT}\n\nUser Query: {prompt}"
        response = st.session_state.chat.send_message(full_message)

        with st.chat_message("assistant"):
            st.markdown(response.text)

        # Save assistant reply
        st.session_state.messages.append({"role": "assistant", "content": response.text})
        # st.session_state.messages.append({"query": prompt, "response": response.text})

        # Optional context viewer
        with st.expander("🔍 See relevant chunks"):
            for i, doc in enumerate(search_results):
                st.markdown(f"**Chunk {i+1}** (Page: {doc.metadata.get('page_label', 'N/A')}):")
                st.write(doc.page_content[:1500] + ("..." if len(doc.page_content) > 1500 else ""))

if st.session_state.messages:
    if st.button("🧹 Reset Chat"):
        del st.session_state.messages

