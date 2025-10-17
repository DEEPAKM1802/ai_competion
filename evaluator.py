# rag_evaluator.py
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def compute_cosine_similarity(query_vector, doc_vectors):
    """Compute cosine similarity between query embedding and document embeddings."""
    sims = cosine_similarity([query_vector], doc_vectors)[0]
    return sims


def recall_at_k(relevant_docs, retrieved_docs, k):
    """Recall@K = (# relevant docs retrieved in top K) / (total relevant docs)."""
    retrieved_k = retrieved_docs[:k]
    intersection = len(set(relevant_docs) & set(retrieved_k))
    total_relevant = len(relevant_docs)
    return intersection / total_relevant if total_relevant > 0 else 0.0


def precision_at_k(relevant_docs, retrieved_docs, k):
    """Precision@K = (# relevant docs retrieved in top K) / K."""
    retrieved_k = retrieved_docs[:k]
    intersection = len(set(relevant_docs) & set(retrieved_k))
    return intersection / k if k > 0 else 0.0


def evaluate_retriever(queries, ground_truth, retriever, embedder, k=5):
    """
    Evaluate retriever performance using Recall@K, Precision@K, and Cosine Similarity.
    
    Args:
        queries (list[str]): List of test queries.
        ground_truth (dict): {query: [relevant_doc_texts]}
        retriever: LangChain retriever (from your backend).
        embedder: Embedding model (OpenAI or HuggingFace).
        k (int): Top-k documents to retrieve.

    Returns:
        dict: Mean metrics across all queries.
    """
    recall_scores, precision_scores, cosine_scores = [], [], []

    for query in queries:
        # 1️⃣ Get relevant docs (truth)
        relevant_texts = ground_truth.get(query, [])
        relevant_set = set(relevant_texts)

        # 2️⃣ Retrieve top-K docs
        results = retriever.invoke(query)
        retrieved_texts = [doc.page_content for doc in results][:k]

        # 3️⃣ Embed query and retrieved docs
        query_vec = embedder.embed_query(query)
        retrieved_vecs = embedder.embed_documents(retrieved_texts)

        # 4️⃣ Compute metrics
        recall = recall_at_k(relevant_set, retrieved_texts, k)
        precision = precision_at_k(relevant_set, retrieved_texts, k)
        cos_sim = np.mean(compute_cosine_similarity(query_vec, retrieved_vecs))

        recall_scores.append(recall)
        precision_scores.append(precision)
        cosine_scores.append(cos_sim)

    return {
        "Recall@K": np.mean(recall_scores),
        "Precision@K": np.mean(precision_scores),
        "CosineSimilarity": np.mean(cosine_scores),
    }



# rag_evaluator.py
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def compute_cosine_similarity(query_vector, doc_vectors):
    """Compute cosine similarity between query embedding and document embeddings."""
    sims = cosine_similarity([query_vector], doc_vectors)[0]
    return sims


def recall_at_k(relevant_docs, retrieved_docs, k):
    """Recall@K = (# relevant docs retrieved in top K) / (total relevant docs)."""
    retrieved_k = retrieved_docs[:k]
    intersection = len(set(relevant_docs) & set(retrieved_k))
    total_relevant = len(relevant_docs)
    return intersection / total_relevant if total_relevant > 0 else 0.0


def precision_at_k(relevant_docs, retrieved_docs, k):
    """Precision@K = (# relevant docs retrieved in top K) / K."""
    retrieved_k = retrieved_docs[:k]
    intersection = len(set(relevant_docs) & set(retrieved_k))
    return intersection / k if k > 0 else 0.0


def evaluate_retriever(queries, ground_truth, retriever, embedder, k=5):
    """
    Evaluate retriever performance using Recall@K, Precision@K, and Cosine Similarity.
    
    Args:
        queries (list[str]): List of test queries.
        ground_truth (dict): {query: [relevant_doc_texts]}
        retriever: LangChain retriever (from your backend).
        embedder: Embedding model (OpenAI or HuggingFace).
        k (int): Top-k documents to retrieve.

    Returns:
        dict: Mean metrics across all queries.
    """
    recall_scores, precision_scores, cosine_scores = [], [], []

    for query in queries:
        # 1️⃣ Get relevant docs (truth)
        relevant_texts = ground_truth.get(query, [])
        relevant_set = set(relevant_texts)

        # 2️⃣ Retrieve top-K docs
        results = retriever.invoke(query)
        retrieved_texts = [doc.page_content for doc in results][:k]

        # 3️⃣ Embed query and retrieved docs
        query_vec = embedder.embed_query(query)
        retrieved_vecs = embedder.embed_documents(retrieved_texts)

        # 4️⃣ Compute metrics
        recall = recall_at_k(relevant_set, retrieved_texts, k)
        precision = precision_at_k(relevant_set, retrieved_texts, k)
        cos_sim = np.mean(compute_cosine_similarity(query_vec, retrieved_vecs))

        recall_scores.append(recall)
        precision_scores.append(precision)
        cosine_scores.append(cos_sim)

    return {
        "Recall@K": np.mean(recall_scores),
        "Precision@K": np.mean(precision_scores),
        "CosineSimilarity": np.mean(cosine_scores),
    }
#=======================================================================================
# rag_evaluator.py

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# ---------- Your custom metrics ----------

def recall_at_k(relevant_set, retrieved_texts, k):
    retrieved_k = retrieved_texts[:k]
    return len(set(relevant_set) & set(retrieved_k)) / len(relevant_set) if relevant_set else 0.0

def precision_at_k(relevant_set, retrieved_texts, k):
    retrieved_k = retrieved_texts[:k]
    return len(set(relevant_set) & set(retrieved_k)) / k if k else 0.0

def average_cosine(query_vec, doc_vecs):
    if len(doc_vecs) == 0:
        return 0.0
    sims = cosine_similarity([query_vec], doc_vecs)[0]
    return float(np.mean(sims))

# ---------- RAGAS metrics integration ----------

from ragas import evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import Faithfulness, FactualCorrectness, ContextRecall, AnswerRelevancy
from ragas.dataset_schema import SingleTurnSample, EvaluationDataset

def evaluate_with_ragas(samples, retriever, embedder, llm, k=5):
    """
    samples: list of dicts, each with:
      {
        "question": str,
        "ground_truths": [str],  # relevant document texts or answer reference
        "answer_reference": str (optional),
      }
    retriever: your retriever (returns docs)
    embedder: embedding model (to compute vectors)
    llm: LLM model for RAGAS (wrapped in LangchainLLMWrapper)
    """
    # Prepare dataset rows for RAGAS
    ragas_samples = []
    for s in samples:
        question = s["question"]
        # fetch context docs
        docs = retriever.get_relevant_documents(question)
        contexts = [d.page_content for d in docs]
        # You may choose only top k
        contexts = contexts[:k]

        # generate answer via your pipeline (or you may already have it)
        # assume you store generated answer in s["generated_answer"]
        answer = s["generated_answer"]

        # ground-truth answer (if available)
        ref = s.get("answer_reference", None)

        ragas_samples.append(
            SingleTurnSample(
                query=question,
                contexts=contexts,
                response=answer,
                reference=ref
            )
        )

    eval_dataset = EvaluationDataset(samples=ragas_samples)

    # Choose metrics
    metrics = [
        Faithfulness(),
        FactualCorrectness(),
        ContextRecall(),
        AnswerRelevancy()
    ]

    # Wrap your LLM in RAGAS wrapper
    evaluator_llm = LangchainLLMWrapper(llm)

    result = evaluate(
        dataset=eval_dataset,
        metrics=metrics,
        llm=evaluator_llm,
        embeddings=None  # can supply embedder if needed
    )

    return result

# ---------- Combined evaluation wrapper ----------

def full_evaluation(samples, retriever, embedder, llm, k=5):
    """
    Runs both custom and RAGAS evaluations.
    """
    custom_metrics = {"Recall@K": [], "Precision@K": [], "CosineSimilarity": []}
    for s in samples:
        question = s["question"]
        relevant_set = set(s.get("ground_truths", []))
        docs = retriever.get_relevant_documents(question)
        retrieved_texts = [d.page_content for d in docs[:k]]

        # compute vectors
        query_vec = embedder.embed_query(question)
        doc_vecs = embedder.embed_documents(retrieved_texts)

        custom_metrics["Recall@K"].append(recall_at_k(relevant_set, retrieved_texts, k))
        custom_metrics["Precision@K"].append(precision_at_k(relevant_set, retrieved_texts, k))
        custom_metrics["CosineSimilarity"].append(average_cosine(query_vec, doc_vecs))

    # Aggregate custom
    custom_report = {
        "Recall@K": float(np.mean(custom_metrics["Recall@K"])),
        "Precision@K": float(np.mean(custom_metrics["Precision@K"])),
        "CosineSimilarity": float(np.mean(custom_metrics["CosineSimilarity"]))
    }

    # RAGAS metrics
    ragas_report = evaluate_with_ragas(samples, retriever, embedder, llm, k=k)

    return {
        "custom": custom_report,
        "ragas": ragas_report
    }
############################
from backend import get_embedding_model, get_retriver, create_vector_store, get_llm
from rag_evaluator import full_evaluation

# Prepare sample test data
samples = [
    {
        "question": "When was X founded?",
        "ground_truths": ["X was founded in 1990"],
        "generated_answer": "X was founded in 1990",
        "answer_reference": "X was founded in 1990"
    },
    # add more
]

embedding_model = get_embedding_model(api_key="YOUR_API")
vectorstore = create_vector_store(docs, embedding_model)
retriever = get_retriver(vectorstore, docs)
llm = get_llm(api_key="YOUR_API")

report = full_evaluation(samples, retriever, embedding_model, llm, k=5)
print(report)
