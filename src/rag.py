"""
RAG pipeline for CrediTrust complaint chatbot.
Uses ChromaDB for retrieval and Flan-T5 for generation.
"""

import chromadb
from chromadb.utils import embedding_functions
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import os

# ----------------------------------------------------------------------
# 1. Load the vector store and embedding function
# ----------------------------------------------------------------------
VECTOR_STORE_PATH = "vector_store"
COLLECTION_NAME = "complaints_full"         # change to "complaints_sample" for testing

embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

client = chromadb.PersistentClient(path=VECTOR_STORE_PATH)

try:
    collection = client.get_collection(name=COLLECTION_NAME, embedding_function=embedding_fn)
except Exception:
    collection = client.get_collection(name="complaints_sample", embedding_function=embedding_fn)

# ----------------------------------------------------------------------
# 2. Load the generator (Flan-T5 base)
# ----------------------------------------------------------------------
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")

def _generate(prompt):
    inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
    outputs = model.generate(inputs.input_ids, max_length=200, num_beams=4, early_stopping=True)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# ----------------------------------------------------------------------
# 3. Prompt template
# ----------------------------------------------------------------------
PROMPT_TEMPLATE = """You are a financial analyst assistant for CrediTrust. 
Your task is to answer questions about customer complaints. 
Use ONLY the following complaint excerpts to formulate your answer. 
If the context does not contain the answer, say that you don't have enough information.

Context:
{context}

Question: {question}

Answer:"""

# ----------------------------------------------------------------------
# 4. Core functions
# ----------------------------------------------------------------------
def retrieve(query: str, k: int = 5):
    query_embedding = embedding_fn([query])
    results = collection.query(query_embeddings=query_embedding, n_results=k)
    chunks = results["documents"][0] if results["documents"] else []
    metadatas = results["metadatas"][0] if results["metadatas"] else []
    return chunks, metadatas

def generate_answer(question: str):
    chunks, metas = retrieve(question)
    if not chunks:
        return "I couldn't find any relevant complaints.", []

    context = "\n\n".join(chunks)
    prompt = PROMPT_TEMPLATE.format(context=context, question=question)
    response = _generate(prompt)

    sources = []
    for i, (chunk, meta) in enumerate(zip(chunks, metas)):
        src = f"Source {i+1}: Complaint ID {meta.get('complaint_id', 'N/A')}\n{chunk[:200]}..."
        sources.append(src)

    return response.strip(), sources