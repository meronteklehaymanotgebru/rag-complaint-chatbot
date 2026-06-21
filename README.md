# RAG Complaint Chatbot – CrediTrust Financial

**Intelligent Complaint Analysis for Financial Services**  
10 Academy · Week 7 Challenge · [Merom T. Gebru](https://github.com/meronteklehaymanotgebru)

---

## 1. Business Context

CrediTrust Financial is a fast‑growing digital finance company serving over **500,000 customers** across East Africa. It offers credit cards, personal loans, savings accounts, and money transfers through a mobile‑first platform. Every month, thousands of unstructured complaints arrive via in‑app channels, email, and regulatory portals. Product managers, support agents, and compliance teams struggle to extract actionable insights from this flood of text.

**Goal:** Build a Retrieval‑Augmented Generation (RAG) chatbot that lets non‑technical users ask plain‑English questions (e.g., *“Why are people unhappy with credit cards?”*) and receive concise, evidence‑backed answers in seconds. The tool must:

- Reduce the time to identify a major complaint trend from **days to minutes**.
- Empower **non‑technical teams** (Support, Compliance) to get answers without a data analyst.
- Shift CrediTrust from **reactive problem‑solving to proactive issue detection**.

---

## 2. Solution Overview

This project delivers a complete RAG pipeline:

1. **Data preprocessing** – Load, clean, and filter CFPB complaint data for four financial products.
2. **Vector search** – Chunk narratives, embed with `all-MiniLM-L6-v2`, and store in **ChromaDB**.
3. **Retrieval & generation** – Retrieve top‑k relevant chunks, inject them into a prompt, and generate an answer using a free LLM (Flan‑T5).
4. **Interactive UI** – A **Gradio** web app with source citations, built for non‑technical users.

All code is version‑controlled, tested, and accompanied by a CI/CD pipeline (GitHub Actions).

---

## 3. Tech Stack

| Component               | Tool / Library                                                                 |
|-------------------------|--------------------------------------------------------------------------------|
| Data manipulation       | `pandas`, `numpy`                                                              |
| EDA & visualisation     | `matplotlib`, `seaborn`                                                        |
| Text chunking           | `langchain_classic.text_splitter`                                              |
| Embedding model         | `sentence-transformers/all-MiniLM-L6-v2`                                       |
| Vector database         | `chromadb` (persistent, with metadata)                                         |
| LLM (generator)         | `google/flan-t5-base` via Hugging Face `pipeline`                              |
| UI                      | `gradio`                                                                       |
| CI/CD                   | GitHub Actions (flake8 + pytest)                                               |
| Environment             | Python 3.10+, `venv`                                                           |

---

## 4. Repository Structure
rag-complaint-chatbot/
├── .github/workflows/ # CI/CD (unittests.yml)
├── data/
│ ├── raw/ # Original CFPB dataset (not committed)
│ └── processed/ # filtered_complaints.csv (not committed)
├── notebooks/
│ └── eda_preprocessing.ipynb # Task 1 – EDA and data cleaning
├── scripts/
│ └── build_vector_store.py # Task 2 – chunking, embedding, ChromaDB indexing
├── src/
│ ├── init.py
│ └── rag.py # Task 3 – RAG pipeline (retriever + generator)
├── tests/
│ ├── init.py
│ └── test_rag.py # Unit tests for RAG functions
├── vector_store/ # Persisted ChromaDB index (sample, not committed)
├── reports/
│ ├── interim_report.md
│ ├── final_report.md
│ └── images/ # EDA and SHAP plots (if applicable)
├── app.py # Task 4 – Gradio interface
├── requirements.txt
├── README.md
└── .gitignore
text


---

## 5. Data

We use the [Consumer Financial Protection Bureau (CFPB) Complaint Database](https://www.consumerfinance.gov/data-research/consumer-complaints/). The dataset contains over 9.6 million complaints; we filter to **Credit Cards, Personal Loans, Savings Accounts, and Money Transfers** with non‑empty narratives. After cleaning, **80,667 credit card complaints** remain. The other three products currently lack narratives in this snapshot, but the pipeline is built to handle them when available.

---

## 6. Setup & Installation

```bash
git clone https://github.com/meronteklehaymanotgebru/rag-complaint-chatbot.git
cd rag-complaint-chatbot
python3 -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt

Requirements (requirements.txt):
text

pandas
matplotlib
seaborn
scikit-learn
langchain-classic
langchain-community
sentence-transformers
chromadb
transformers
gradio
torch
jupyter
pytest
flake8
tqdm

7. Running the Project Step‑by‑Step
Task 1 – EDA & Data Preprocessing

    Place the raw CFPB CSV (consumer_complaints.csv) inside data/raw/.

    Open and run the notebook notebooks/eda_preprocessing.ipynb.
    It loads the full dataset, performs EDA, filters to the four target products, cleans narratives, and saves data/processed/filtered_complaints.csv.

Task 2 – Vector Store (Sampled)

Execute the standalone script:
bash

python scripts/build_vector_store.py

It samples 500 complaints, chunks them, generates embeddings, and persists a ChromaDB index in vector_store/.
For the full RAG pipeline, we switch to the pre‑built vector store (1.37M chunks) provided by the challenge organisers.
Task 3 – RAG Pipeline

The core logic lives in src/rag.py. It:

    Loads the pre‑built ChromaDB collection.

    Embeds a user question and retrieves the top‑k (default 5) relevant chunks.

    Injects the chunks into a prompt template and generates an answer using Flan‑T5.

Example usage:
python

from src.rag import generate_answer
answer, sources = generate_answer("What are the main credit card complaints?")
print(answer)

Task 4 – Gradio App

Launch the interactive UI:
bash

python app.py

This starts a Gradio server at http://127.0.0.1:7860. Users can type questions, see AI‑generated answers, and inspect the source complaint snippets below the answer.
8. Evaluation

We qualitatively evaluated the system using 10 representative questions. Each answer was scored on a 1‑5 scale based on relevance, factual grounding, and clarity. The evaluation table (with sample questions, generated answers, retrieved sources, and scores) is included in the final report.
9. CI/CD Pipeline

On every push and pull request to main, GitHub Actions runs:

    flake8 – code style checks (src/, scripts/, tests/).

    pytest – unit tests in tests/.

The workflow file is at .github/workflows/unittests.yml. The build fails if linting errors or test failures are detected.
10. Development Workflow

We followed a branch‑per‑task strategy with Pull Requests:

    task-1 – EDA, data cleaning, and preprocessing.

    task-2 – Text chunking, embedding, and vector store indexing.

    task-3 – RAG core logic, prompt engineering, and evaluation.

    task-4 – Gradio UI and final integration.

Each branch was merged into main via Pull Request, with descriptive Conventional Commits (e.g., feat: add chunking script, fix: resolve ChromaDB import error). This creates a clean, auditable history.
11. Limitations & Future Work

    Data coverage: Only Credit Card complaints currently have narratives; the other three products are not yet represented. As more narrative data becomes available, the pipeline can ingest it with no code changes.

    Hardware constraints: The sampled vector store uses only 500 complaints. For production, we rely on the pre‑built full‑scale vector store (1.37M chunks) provided by the challenge.

    LLM performance: Flan‑T5‑base is a small model that runs on CPU. For better answer quality, it can be swapped with a larger model (e.g., Llama 2, Mistral) or a paid API.

    Streaming: The Gradio app does not yet stream responses; this can be added using Hugging Face’s TextStreamer.

    Productionisation: Future work could containerise the app with Docker, add authentication, and integrate with CrediTrust’s internal systems.

12. References

    CFPB Consumer Complaint Database

    LangChain Documentation

    ChromaDB Documentation

    Sentence‑Transformers

    Hugging Face Flan‑T5

    Gradio Documentation


