# core/UI.py
import streamlit as st
from backend import rag_setup, query_pipeline

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(page_title="Enterprise RAG Chat", page_icon="💬", layout="wide")
st.title("💬 Enterprise RAG Chatbot")
st.markdown("Chat naturally with your document-powered AI assistant.")

# -----------------------------
# SESSION STATE
# -----------------------------
if "session_id" not in st.session_state:
    st.session_state.session_id = "user_001"

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# -----------------------------
# SIDEBAR — FILE INGESTION
# -----------------------------
with st.sidebar:
    st.header("📄 Upload Knowledge Base")
    api_key = st.text_input("🔑 Google API Key", type="password")

# -----------------------------
# CHAT INTERFACE
# -----------------------------
st.markdown("---")
st.subheader("💬 Chat Interface")

user_input = st.chat_input("Ask something about your document...", accept_file= True)

# if user_input and api_key:
#     if user_input['text']:
#         if user_input['files']:
#             with st.spinner("Processing file..."):
#                 rag_setup(api_key, scenario="ingest", file_path=f"data/{user_input['files'][0].name}")
#             st.success("✅ File processed and stored in vector database!")
#         with st.spinner("Thinking..."):
#             response = query_pipeline(api_key, user_input['text'])
#     answer = response["answer"]
#     st.session_state.chat_history = response["chat_history"]

if user_input and api_key:
    with st.spinner("Processing..."):
        response = query_pipeline(api_key, user_input, session_id=st.session_state.session_id)
    answer = response["answer"]
    st.session_state.chat_history = response["chat_history"]

# Display chat history
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


