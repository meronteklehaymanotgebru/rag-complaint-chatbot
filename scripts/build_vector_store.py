"""
Task 2 – Text Chunking, Embedding, and Vector Store Indexing
Processes a small stratified sample of complaints, chunks them,
generates embeddings, and stores them in a ChromaDB vector store.
"""

import pandas as pd
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
import chromadb
from chromadb.utils import embedding_functions
import os
from tqdm import tqdm

# -------------------------------------------------------------------
# 1. Load filtered data
# -------------------------------------------------------------------
print("Loading filtered complaints...")
df = pd.read_csv('data/processed/filtered_complaints.csv')
print(f"Total complaints available: {len(df)}")

# -------------------------------------------------------------------
# 2. Create a stratified sample (tiny – to respect hardware limits)
# -------------------------------------------------------------------
# Since only Credit card has data, we still do it for demonstration.
sample_size = 500   # keep it very small
df_sample = df.sample(n=500, random_state=42).reset_index(drop=True)

print(f"Sample size: {len(df_sample)}")
print(df_sample['Product'].value_counts())

# -------------------------------------------------------------------
# 3. Text chunking
# -------------------------------------------------------------------
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separators=["\n\n", "\n", ".", " ", ""]
)

documents = []
metadata = []
ids = []

for idx, row in tqdm(df_sample.iterrows(), total=len(df_sample)):
    chunks = splitter.split_text(row['narrative'])
    for i, chunk in enumerate(chunks):
        documents.append(chunk)
        metadata.append({
            'product': row['Product'],
            'complaint_id': str(row['Complaint ID']),
            'issue': row['Issue'] if pd.notna(row['Issue']) else '',
            'company': row['Company'] if pd.notna(row['Company']) else '',
            'chunk_index': i
        })
        ids.append(f"{row['Complaint ID']}_{i}")

print(f"Total chunks: {len(documents)}")

# -------------------------------------------------------------------
# 4. Create ChromaDB collection with the embedding model
# -------------------------------------------------------------------
# Use all-MiniLM-L6-v2 via Chroma's built-in embedding function
embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name='all-MiniLM-L6-v2'
)

# Persist the vector store in the vector_store/ directory
os.makedirs('vector_store', exist_ok=True)
client = chromadb.PersistentClient(path='vector_store')

collection = client.get_or_create_collection(
    name='complaints_sample',
    embedding_function=embedding_fn
)

# Add documents in batches to avoid memory issues
batch_size = 100
for i in tqdm(range(0, len(documents), batch_size)):
    batch_docs = documents[i:i+batch_size]
    batch_meta = metadata[i:i+batch_size]
    batch_ids = ids[i:i+batch_size]
    collection.add(
        documents=batch_docs,
        metadatas=batch_meta,
        ids=batch_ids
    )

print(f"Vector store saved to 'vector_store/' with {collection.count()} chunks.")