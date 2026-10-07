import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

import os
import time
import base64
import io
import streamlit as st
from dotenv import load_dotenv
import pymupdf as fitz
from PIL import Image
from openai import OpenAI  # Native lightweight client for fast streaming and vision calls

from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings, ChatNVIDIA
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate

# Compatibility import for LangChain chains
try:
    from langchain_classic.chains.combine_documents import create_stuff_documents_chain
    from langchain_classic.chains import create_retrieval_chain
except ModuleNotFoundError:
    from langchain.chains.combine_documents import create_stuff_documents_chain
    from langchain.chains import create_retrieval_chain

# 1. Environment & API Key Setup
load_dotenv()

api_key = os.getenv("NVIDIA_API_KEY") or os.getenv("api") or os.getenv("API_KEY") or ""
os.environ["NVIDIA_API_KEY"] = api_key

BASE_URL = "https://integrate.api.nvidia.com/v1"
MODEL_NAME = "deepseek-ai/deepseek-v4.1-flash"

# Page Configuration
st.set_page_config(page_title="NVIDIA NIM Multi-Engine", layout="wide")
st.title("NVIDIA NIM Multimodal & Chat Suite")

# Sidebar Mode Switcher
mode = st.sidebar.radio(
    "Select Operating Mode:", 
    [
        "Text RAG (FAISS Document Search)", 
        "Direct Visual Analysis (PNG, JPG, PDF)", 
        "General LLM Chat (With Memory)"
    ]
)

# Helper function to downsample and compress images for low payload latency
def process_and_compress_image(image_bytes, max_size=(1024, 1024)):
    img = Image.open(io.BytesIO(image_bytes))
    if img.mode != "RGB":
        img = img.convert("RGB")
    img.thumbnail(max_size, Image.Resampling.LANCZOS)
    
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=85)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")

# Initialize LangChain model for RAG
deepseek_llm = ChatNVIDIA(
    base_url=BASE_URL,
    model=MODEL_NAME,
    nvidia_api_key=api_key,
    timeout=180
)

# Native OpenAI Client for Visual Analysis and General Chat
native_client = OpenAI(
    base_url=BASE_URL,
    api_key=api_key,
    timeout=180.0
)

# =====================================================================
# MODE 1: TEXT RAG SEARCH
# =====================================================================
if mode == "Text RAG (FAISS Document Search)":
    st.subheader("Text-Based RAG Search over Uploaded Documents")

    os.makedirs("./us_census", exist_ok=True)
    rag_files = st.file_uploader(
        "Upload PDF documents for RAG indexing:", 
        type=["pdf"], 
        accept_multiple_files=True
    )

    if rag_files:
        for file in rag_files:
            file_path = os.path.join("./us_census", file.name)
            with open(file_path, "wb") as f:
                f.write(file.getbuffer())
        st.success(f"Saved {len(rag_files)} file(s) to `./us_census`.")

    def vector_embeddings():
        with st.spinner("Ingesting documents from ./us_census..."):
            st.session_state.embeddings = NVIDIAEmbeddings(
                base_url=BASE_URL,
                model="deepseek-ai/deepseek-v4.1-flash",
                nvidia_api_key=api_key
            )
            st.session_state.loader = PyPDFDirectoryLoader("./us_census")
            st.session_state.docs = st.session_state.loader.load()
            
            if not st.session_state.docs:
                st.error("No valid PDF documents found in `./us_census`. Please upload a PDF first.")
                return

            st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            st.session_state.final_document = st.session_state.text_splitter.split_documents(st.session_state.docs)

            st.session_state.vectors = FAISS.from_documents(st.session_state.final_document, st.session_state.embeddings)
            st.success(f"Vector store ready! Loaded {len(st.session_state.final_document)} chunks.")

    prompt = ChatPromptTemplate.from_template(
    """
    Answer the question based on the provided context only.
    Please provide the most accurate response based on the question.

    <context>
    {context}
    </context>

    Question: {input}
    """
    )

    if st.button("Ingest Document Embeddings"):
        vector_embeddings()

    user_question = st.text_input("Enter Your Question From Documents:")

    if user_question:
        if "vectors" not in st.session_state:
            st.error("Please click 'Ingest Document Embeddings' button first to build the vector store!")
        else:
            try:
                document_chain = create_stuff_documents_chain(deepseek_llm, prompt)
                retriever = st.session_state.vectors.as_retriever(search_kwargs={"k": 4})
                retrieval_chain = create_retrieval_chain(retriever, document_chain)

                start_time = time.process_time()
                response = retrieval_chain.invoke({'input': user_question})
                elapsed_time = time.process_time() - start_time

                st.write(f"**Response from DeepSeek Flash (retrieved in {elapsed_time:.2f}s):**")
                st.write(response["answer"])

                with st.expander("Document Similarity Search (Retrieved Chunks)"):
                    for i, doc in enumerate(response["context"]):
                        st.markdown(f"**Chunk {i+1}:**")
                        st.write(doc.page_content)
                        st.caption(f"Source: {doc.metadata.get('source', 'Unknown')}")
                        st.divider()
            except Exception as e:
                st.error(f"API request failed: {str(e)}")

# =====================================================================
# MODE 2: DIRECT VISUAL ANALYSIS (OPTIMIZED NATIVE STREAMING)
# =====================================================================
elif mode == "Direct Visual Analysis (PNG, JPG, PDF)":
    st.subheader("Direct Multimodal Analysis (Images & PDF Visuals)")

    uploaded_file = st.file_uploader(
        "Upload a PNG, JPG image or a PDF document:", 
        type=["png", "jpg", "jpeg", "pdf"]
    )

    if uploaded_file:
        base64_image = None

        if uploaded_file.type in ["image/png", "image/jpeg", "image/jpg"]:
            image_bytes = uploaded_file.read()
            base64_image = process_and_compress_image(image_bytes)
            st.image(image_bytes, caption="Uploaded Image Preview", width=450)

        elif uploaded_file.type == "application/pdf":
            pdf_doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
            page = pdf_doc.load_page(0)
            pix = page.get_pixmap(dpi=150)
            image_bytes = pix.tobytes("png")
            base64_image = process_and_compress_image(image_bytes)
            st.image(image_bytes, caption="PDF Page 1 Rendered Image", width=450)

        visual_prompt = st.text_input("Ask a question about this visual/image (Press Enter):")

        if visual_prompt:
            with st.chat_message("assistant"):
                try:
                    completion = native_client.chat.completions.create(
                        model=MODEL_NAME,
                        messages=[
                            {
                                "role": "user",
                                "content": [
                                    {"type": "text", "text": visual_prompt},
                                    {
                                        "type": "image_url",
                                        "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                                    }
                                ]
                            }
                        ],
                        temperature=0.7,
                        max_tokens=2048,
                        stream=True
                    )

                    def get_visual_stream():
                        for chunk in completion:
                            if chunk.choices and chunk.choices[0].delta.content:
                                yield chunk.choices[0].delta.content

                    st.write_stream(get_visual_stream())

                except Exception as e:
                    st.error(f"Visual analysis failed: {str(e)}")

# =====================================================================
# MODE 3: GENERAL CHAT WITH MEMORY
# =====================================================================
else:
    st.subheader("General Conversational Chat")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    if st.sidebar.button("Clear Chat History"):
        st.session_state.chat_history = []
        st.rerun()

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    user_input = st.chat_input("Type your message here...")

    if user_input:
        with st.chat_message("user"):
            st.write(user_input)
        st.session_state.chat_history.append({"role": "user", "content": user_input})

        with st.chat_message("assistant"):
            try:
                recent_messages = st.session_state.chat_history[-6:]
                formatted_messages = [
                    {"role": m["role"], "content": m["content"]} for m in recent_messages
                ]

                completion = native_client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=formatted_messages,
                    temperature=0.7,
                    max_tokens=2048,
                    stream=True
                )

                def get_stream():
                    for chunk in completion:
                        if chunk.choices and chunk.choices[0].delta.content:
                            yield chunk.choices[0].delta.content

                full_response = st.write_stream(get_stream())
                st.session_state.chat_history.append({"role": "assistant", "content": full_response})

            except Exception as e:
                st.error(f"Chat request failed: {str(e)}")