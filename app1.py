import os
import re
import fitz  # PyMuPDF
import streamlit as st
from dotenv import load_dotenv
from datetime import datetime

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

# --- Load environment variables ---
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# --- API Validation ---
if not OPENAI_API_KEY:
    st.error("❌ Missing OpenAI API key in your .env file!")
    st.stop()

# --- Streamlit Config ---
st.set_page_config(page_title="📄 Multi-PDF Chatbot with Context")

# --- Cached resources ---
@st.cache_resource
def get_embeddings():
    return OpenAIEmbeddings()

@st.cache_resource
def get_llm():
    return ChatOpenAI(model="gpt-4o", temperature=1)

# --- Initialize FAISS vectorstore ---
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

# --- Extract text from PDF ---
def extract_text(file):
    try:
        pdf = fitz.open(stream=file.read(), filetype="pdf")
        return "\n".join([page.get_text() for page in pdf])
    except Exception as e:
        st.error(f"❌ Failed to extract text: {e}")
        return ""

# --- Create chunks and build vectorstore ---
def create_vectorstore_from_text(text, source_name):
    doc = Document(page_content=text, metadata={"source": source_name})
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents([doc])
    return FAISS.from_documents(chunks, get_embeddings())

def update_vectorstore(text, source_name):
    new_vs = create_vectorstore_from_text(text, source_name)
    if st.session_state.vectorstore is None:
        st.session_state.vectorstore = new_vs
    else:
        st.session_state.vectorstore.merge_from(new_vs)

# --- Setup QA chain ---
def get_qa_chain():
    return RetrievalQA.from_chain_type(
        llm=get_llm(),
        retriever=st.session_state.vectorstore.as_retriever(search_kwargs={"k": 5}),
        chain_type="stuff"
    )

# --- Objection detection using regex ---
# def detect_objection(text):
#     objection_patterns = [
#         r"too expensive", r"not interested", r"already have", r"need to discuss", r"maybe later",
#         r"not the right time", r"no budget", r"don’t trust", r"prefer someone else", r"have to think"
#     ]
#     for pattern in objection_patterns:
#         if re.search(pattern, text.lower()):
#             return True
#     return False

# # --- Block off-topic queries ---
# def is_off_topic(query):
#     off_topic_keywords = ["weather", "sports", "politics", "movies", "actor", "game", "instagram", "twitter","country","political issue"]
#     return any(keyword in query.lower() for keyword in off_topic_keywords)

# --- Log chats ---
def log_chat(user_q, bot_a):
    os.makedirs("logs", exist_ok=True)
    with open("logs/chat_logs.txt", "a", encoding="utf-8") as f:
        f.write(f"{datetime.now()} | USER: {user_q.strip()}\n")
        f.write(f"{datetime.now()} | BOT: {bot_a.strip()}\n\n")

# --- UI: Title and Upload ---
st.title("🧠 Context-Aware Multi-PDF Chatbot")

uploaded_files = st.sidebar.file_uploader("📄 Upload one or more PDFs", type=["pdf"], accept_multiple_files=True)

if uploaded_files:
    for uploaded_file in uploaded_files:
        content = extract_text(uploaded_file)
        if content:
            update_vectorstore(content, uploaded_file.name)
            st.sidebar.success(f"✅ Uploaded and embedded: {uploaded_file.name}")

# --- Chat handler ---
if st.session_state.vectorstore:
    qa = get_qa_chain()

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    user_input = st.chat_input("💬 Ask something about your PDFs...")
    if user_input:
        # st.chat_message("user").markdown(user_input)
        response = qa.run(user_input)
        # # --- Handle off-topic ---
        # if is_off_topic(user_input):
        #     response = "🚫 That question seems off-topic. Please ask something related to the uploaded PDFs."
        # else:
        #     response = qa.run(user_input)

        #     # --- Objection Detection ---
        #     if detect_objection(user_input):
        #         response += "\n\n⚠️ Detected a potential objection. You might want to address the customer's concern!"

        # Log and display
        log_chat(user_input, response)
        # st.chat_message("assistant").markdown(response)
        st.session_state.chat_history.append((user_input, response))

    # Display past chat history
    for user_msg, bot_msg in st.session_state.chat_history:
        with st.chat_message("user"):
            st.markdown(user_msg)
        with st.chat_message("assistant"):
            st.markdown(bot_msg)
else:
    st.info("📥 Please upload one or more PDFs to begin.")
