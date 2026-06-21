"""
Gradio web interface for the CrediTrust Complaint Chatbot.
"""

import gradio as gr
from src.rag import generate_answer

def ask_question(question):
    if not question.strip():
        return "Please enter a question.", ""
    answer, sources = generate_answer(question)
    source_text = "\n\n".join(sources)
    return answer, source_text

with gr.Blocks(title="CrediTrust Complaint Analyst") as demo:
    gr.Markdown("# 💬 CrediTrust Complaint Analyst")
    gr.Markdown("Ask any question about customer complaints and get evidence‑backed answers.")

    with gr.Row():
        with gr.Column(scale=3):
            question_input = gr.Textbox(
                label="Your question",
                placeholder="e.g., Why are people unhappy with Credit Cards?",
                lines=2
            )
            submit_btn = gr.Button("Ask", variant="primary")
            clear_btn = gr.ClearButton(value="Clear")

        with gr.Column(scale=2):
            answer_output = gr.Textbox(label="AI Answer", lines=8, interactive=False)

    with gr.Accordion("📄 Source Complaints", open=False):
        sources_output = gr.Markdown()

    submit_btn.click(
        fn=ask_question,
        inputs=question_input,
        outputs=[answer_output, sources_output]
    )
    clear_btn.add([question_input, answer_output, sources_output])

if __name__ == "__main__":
    demo.launch()