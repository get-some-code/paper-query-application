import os
import re
import fitz
import time
import random
from qdrant_client.http import models
import tempfile
import streamlit as st
from dotenv import load_dotenv
from langchain.schema import Document
from langchain_community.vectorstores import Qdrant as QdrantVectorStore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from qdrant_client import QdrantClient
import google.generativeai as genai
from langchain_community.document_loaders import PyPDFLoader

# Load API key
load_dotenv()
GOOGLE_API_KEY = os.getenv("GEMINI_API_KEY")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)

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

# Helper function: embed text with retry
def embed_with_retry(embedding_model, text, max_retries=3):
    for attempt in range(max_retries):
        try:
            return embedding_model.embed_query(text)
        except Exception as e:
            print(f"Embedding attempt {attempt+1} failed: {e}")
            time.sleep((2 ** attempt) + random.random())
    raise Exception("Embedding failed after retries")

# Process PDF
if file and st.button("📚 Prepare PDF"):
    with st.spinner("Please wait. Your PDF is being processed..."):
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

    # Embeddings and Qdrant setup
    embedding_model = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=GOOGLE_API_KEY
    )
    qdrant_client = QdrantClient(
        url="https://3e3583cf-d0e6-4cc1-b344-1dec8fdce1cc.us-east4-0.gcp.cloud.qdrant.io", 
        api_key=QDRANT_API_KEY,
        prefer_grpc=False
    )

    collection_name = "rag-chat-application"

    # Delete existing collection to recreate fresh index
    if qdrant_client.collection_exists(collection_name):
        qdrant_client.delete_collection(collection_name)

    # Create collection with vector size inferred from embedding a dummy string
    vector_size = len(embedding_model.embed_query("test"))
    qdrant_client.create_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(size=vector_size, distance=models.Distance.COSINE)
    )

    # Embed and upsert chunks one by one with retry and slight delay
    with st.spinner("Vectorizing and indexing your PDF..."):
        for idx, chunk in enumerate(chunks):
            vector = embed_with_retry(embedding_model, chunk.page_content)
            qdrant_client.upsert(
                collection_name=collection_name,
                points=[
                    models.PointStruct(
                        id=idx,
                        vector=vector,
                        payload={
                            "text": chunk.page_content,
                            **chunk.metadata
                        }
                    )
                ]
            )
            st.write(f"Indexed chunk {idx+1} of {len(chunks)}")
            time.sleep(0.2)  # small delay to avoid rate limits

    st.success("✅ Successfully Processed and Indexed PDF!")

    # Save session state
    st.session_state.pdf_ready = True
    st.session_state.embedding_model = embedding_model
    st.session_state.qdrant_client = qdrant_client
    st.session_state.collection_name = collection_name
    st.session_state.model = genai.GenerativeModel(model_name="gemini-1.5-flash")
    st.session_state.chat = st.session_state.model.start_chat()

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

        # Similarity search in Qdrant
        search_results = st.session_state.qdrant_client.search(
            collection_name=st.session_state.collection_name,
            query_vector=st.session_state.embedding_model.embed_query(prompt),
            limit=3,
            with_payload=True
        )

        # Prepare context from payloads
        context = "\n\n\n".join([
            f"Contents: {hit.payload.get('text','')[:1000]}...\nPage: {hit.payload.get('page_label','N/A')}\nSource: {hit.payload.get('source','N/A')}"
            for hit in search_results
        ])

        SYSTEM_PROMPT = f"""
        You are a helpful AI Assistant who answers user queries based ONLY on the context retrieved from a PDF file.
        Always mention the page number and keep your answers precise and helpful.

        Context:
        {context}
        """

        full_message = f"{SYSTEM_PROMPT}\n\nUser Query: {prompt}"
        response = st.session_state.chat.send_message(full_message)

        with st.chat_message("assistant"):
            st.markdown(response.text)

        # Save assistant reply
        st.session_state.messages.append({"role": "assistant", "content": response.text})

        # Optional context viewer
        with st.expander("🔍 See relevant chunks"):
            for i, hit in enumerate(search_results):
                text = hit.payload.get("text", "")
                page = hit.payload.get("page_label", "N/A")
                st.markdown(f"**Chunk {i+1}** (Page: {page}):")
                st.write(text[:1500] + ("..." if len(text) > 1500 else ""))

# Reset Chat button
if st.session_state.messages:
    if st.button("🧹 Reset Chat"):
        del st.session_state.messages
        st.experimental_rerun()
