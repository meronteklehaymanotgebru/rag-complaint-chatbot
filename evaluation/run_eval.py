"""Run the RAG pipeline on a list of questions and record outputs."""
import csv, json
from src.rag import generate_answer

with open('evaluation/eval_questions.csv', 'r') as f:
    questions = [row['question'] for row in csv.DictReader(f)]

results = []
for q in questions:
    answer, sources = generate_answer(q)
    results.append({
        'question': q,
        'answer': answer,
        'source_1': sources[0] if sources else '',
        'source_2': sources[1] if len(sources) > 1 else ''
    })
    print(f"Q: {q}\nA: {answer}\n")

# Save to JSON for later use
with open('evaluation/eval_results.json', 'w') as f:
    json.dump(results, f, indent=2)