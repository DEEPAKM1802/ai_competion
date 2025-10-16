# backend.py
from langchain.chains import create_retrieval_chain, ConversationalRetrievalChain
from langchain.chains.combine_documents import create_stuff_documents_chain

from langchain_community.retrievers import BM25Retriever
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_community.document_loaders import PyPDFLoader

from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain.retrievers import EnsembleRetriever, ContextualCompressionRetriever

from langchain.prompts import PromptTemplate
from langchain.schema import Document
from langchain.memory import ConversationBufferMemory

from langchain_experimental.text_splitter import SemanticChunker
from langchain.text_splitter import TokenTextSplitter

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

import math
import os

# ==========================================================
# 1. FILE INGESTION
# ==========================================================
def load_data(file_path):
    """Load documents from PDF or text file."""
    if file_path.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
    else:
        raise ValueError("Unsupported file type. Please upload PDF or TXT.")
    return loader.load()

# ==========================================================
# 2. CHUNKING / PREPROCESSING
# ==========================================================
def split_documents(docs, chunk_size=500, chunk_overlap=100):
    """
    Semantic chunking + LangChain tokenizer + metadata tagging.
    """
    embed_model = HuggingFaceEmbeddings(model_name="all-mpnet-base-v2")

    semantic_chunker = SemanticChunker(embed_model, breakpoint_threshold_type="percentile", breakpoint_threshold_amount=95, number_of_chunks=10)
    semantic_chunks = semantic_chunker.split_documents([d for d in docs])

    # 4️⃣ Initialize LangChain tokenizer
    tokenizer = TokenTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    # 5️⃣ Add metadata and token counts
    processed_chunks = []
    for i, chunk in enumerate(semantic_chunks):
        text = chunk.page_content.strip()

        # Tokenize text
        tokens = tokenizer._tokenizer.encode(text)
        token_count = len(tokens)

        # Estimate lines
        total_lines = text.count("\n") + 1
        mid_line = math.ceil(total_lines / 2)

        # Enrich metadata
        metadata = {
            **chunk.metadata,
            "chunk_id": f"chunk_{i}",
            "token_count": token_count,
            "source_id": chunk.metadata.get("source", f"doc_{i}"),
            "start_line_est": max(1, mid_line - 1),
            "end_line_est": mid_line + 1,
        }

        processed_chunks.append(Document(page_content=text, metadata=metadata))

    return processed_chunks

# ==========================================================
# 3. EMBEDDING INITIALIZATION
# ==========================================================
def get_embedding_model(api_key, model_name="all-mpnet-base-v2"):
    """Initialize embedding model (Google GenAI or any other)."""
    from langchain_community.embeddings import SentenceTransformerEmbeddings
    embedding_model = HuggingFaceEmbeddings(model_name="all-mpnet-base-v2",  encode_kwargs={"normalize_embeddings": True, "batch_size": 100})
    return embedding_model

# ==========================================================
# 4. VECTOR STORE CREATION
# ==========================================================
def create_vector_store(chunks, embedding_model, persist_dir="data/chroma"):
    """Create and persist vector store using Chroma."""
    os.makedirs(persist_dir, exist_ok=True)
    vectorstore = Chroma.from_documents(chunks, embedding_model, persist_directory=persist_dir)
    return vectorstore

# ==========================================================
# 5. RETRIEVER + RE-RANKER
# ==========================================================
def get_retriver(vectorstore, docs, top_k=5):
    """
    Create 3 retriever scenarios:
        1️⃣ Vector retriever (cosine similarity)
        2️⃣ BM25 retriever (lexical)
        3️⃣ Hybrid (vector + BM25 ensemble)
    Optionally re-rank results using a cross-encoder.
    """

    # --- (A) VECTOR RETRIEVER ---
    vector_retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 10, "fetch_k": 50, "lambda_mult": 0.2}
    )

    # --- (B) BM25 RETRIEVER ---
    bm25_retriever = BM25Retriever.from_documents(docs)
    bm25_retriever.k = top_k

    # --- (C) HYBRID RETRIEVER (Vector + BM25) ---
    hybrid_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, vector_retriever],
        weights=[0.2, 0.8]
    )

    # --- (D) RE-RANKER (CrossEncoder) ---
    cross_encoder = HuggingFaceCrossEncoder(model_name="cross-encoder/stsb-roberta-base") #"cross-encoder/ms-marco-MiniLM-L-6-v2"
    reranker = CrossEncoderReranker(model=cross_encoder, top_n=top_k)

    # Use ContextualCompressionRetriever to apply reranker
    reranked_vector = ContextualCompressionRetriever(
        base_compressor=reranker, base_retriever=vector_retriever
    )

    reranked_bm25 = ContextualCompressionRetriever(
        base_compressor=reranker, base_retriever=bm25_retriever
    )

    reranked_hybrid = ContextualCompressionRetriever(
        base_compressor=reranker, base_retriever=hybrid_retriever
    )

    # Wrap the retrievers with re-ranker option
    retriever_scenarios = {
        "vector | re_rank=False": vector_retriever,
        "bm25 | re_rank=False": bm25_retriever,
        "hybrid | re_rank=False": hybrid_retriever,

        "vector | re_rank=True":  reranked_vector,
        "bm25 | re_rank=True":    reranked_bm25,
        "hybrid | re_rank=True":  reranked_hybrid,
    }

    # --- (G) Default Retriever: Vector with Ranker ---
    default_retriever = retriever_scenarios["vector | re_rank=True"]

    return default_retriever

# ==========================================================
# 6. LLM INITIALIZATION
# ==========================================================
def get_llm(api_key, model_name="gemini-2.5-flash", temperature=0.3, max_output_tokens=512, top_p=0.9, top_k=40):
    """Initialize the LLM."""
    return ChatGoogleGenerativeAI(
        model=model_name,
        google_api_key=api_key,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
        top_p=top_p,
        top_k=top_k,
    )


# ==========================================================
# 7. PROMPT TEMPLATE
# ==========================================================
def get_prompt_template():
    """Return the QA prompt template."""
    return PromptTemplate(
        template=(
            "You are a helpful AI assistant.\n"
            "Use the following context to answer the user's question accurately.\n\n"
            "Context:\n{context}\n\n"
            "Question:\n{question}\n\n"
            "Answer:"
        ),
        input_variables=["context", "question"],
    )


# ==========================================================
# 8. CREATE RETRIEVAL CHAIN
# ==========================================================
def create_rag_chain(llm, retriever, prompt_template):
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer"  # ✅ Explicitly tell memory what to store
    )

    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        combine_docs_chain_kwargs={"prompt": prompt_template},
        return_source_documents=True,
        output_key="answer"  # ✅ Tell the chain which output to use as final answer
    )

    return chain



# ==========================================================
# 9. SCENARIO 1: INGESTION PIPELINE
# ==========================================================
def ingest_pipeline(api_key, file_path, chunk_size=500, chunk_overlap=100):
    """Pipeline for file upload -> chunk -> embedding -> vector DB."""

    docs = load_data(file_path)

    chunks = split_documents(docs, chunk_size, chunk_overlap)

    embedding_model = get_embedding_model(api_key)

    vectorstore = create_vector_store(chunks, embedding_model)

    return vectorstore


# ==========================================================
# 10. SCENARIO 2: QUERY PIPELINE
# ==========================================================
chat_sessions = {}
def query_pipeline(api_key, user_input, session_id="default", vectorstore_path="data/chroma", model_name="gemini-2.5-flash"):
    """
    Unified RAG pipeline:
    - If user uploads a file, run ingestion + vectorstore update
    - If user only sends a question, perform retrieval + LLM generation
    """

    # Ensure vectorstore directory
    os.makedirs(vectorstore_path, exist_ok=True)

    embedding_model = get_embedding_model(api_key)
    llm = get_llm(api_key, model_name=model_name)
    prompt_template = get_prompt_template()

    # 🧩 Step 1 — Ingest new document (if file is uploaded)
    if user_input.get("files"):
        file_obj = user_input["files"][0]
        file_path = f"{vectorstore_path}/{file_obj.name}"
        file_obj.save(file_path)
        docs = load_data(file_path)
        chunks = split_documents(docs)
        vectorstore = create_vector_store(chunks, embedding_model, persist_dir=vectorstore_path)
    else:
        vectorstore = Chroma(persist_directory=vectorstore_path, embedding_function=embedding_model)

    # 🧩 Step 2 — Create retriever
    docs_data = vectorstore.get(include=["metadatas", "documents"])
    docs = [Document(page_content=d, metadata=m) for d, m in zip(docs_data["documents"], docs_data["metadatas"])]
    retriever = get_retriver(vectorstore, docs)

    # 🧩 Step 3 — Load or create RAG chain with memory
    if session_id not in chat_sessions:
        chat_sessions[session_id] = create_rag_chain(llm, retriever, prompt_template)

    rag_chain = chat_sessions[session_id]

    # 🧩 Step 4 — Run query
    prompt = user_input["text"]
    response = rag_chain.invoke({"question": prompt})  # or {"input": prompt} if you chose Option B
    answer = response.get("answer") or response.get("result", "")

    chat_history = [
        {"role": "user", "content": msg.content} if msg.type == "human" else {"role": "assistant", "content": msg.content}
        for msg in rag_chain.memory.chat_memory.messages
    ]

    return {"answer": answer, "chat_history": chat_history}

# ==========================================================
# 11. MASTER FUNCTION
# ==========================================================
def rag_setup(api_key, scenario, **kwargs):
    """Main orchestrator function for both scenarios."""
    if scenario == "ingest":
        return ingest_pipeline(api_key, kwargs["file_path"])
    elif scenario == "query":
        return query_pipeline(api_key, kwargs["prompt"])
    else:
        raise ValueError("Invalid scenario. Use 'ingest' or 'query'.")


