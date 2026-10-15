"""Web app cho Chatbot Sổ tay sinh viên.

Chỉ nạp artifact RAG đã được notebook tạo; vì vậy không cần Jupyter hay
pdfplumber khi khởi động trên Render.
"""

import os
from pathlib import Path

import gradio as gr
import joblib
from sklearn.metrics.pairwise import cosine_similarity


BASE_DIR = Path(__file__).resolve().parent
ARTIFACT_PATH = BASE_DIR / "artifacts" / "rag_index.joblib"
MIN_RELEVANCE = 0.055


def load_rag_index():
    if not ARTIFACT_PATH.exists():
        raise FileNotFoundError(
            f"Không tìm thấy {ARTIFACT_PATH}. Hãy chạy cell 'Xuất chỉ mục' "
            "trong student_handbook_rag.ipynb rồi commit thư mục artifacts/."
        )
    return joblib.load(ARTIFACT_PATH)


INDEX = load_rag_index()
VECTORIZER = INDEX["vectorizer"]
MATRIX = INDEX["matrix"]
DOCUMENTS = INDEX["documents"]


def answer_from_handbook(question: str) -> str:
    question = " ".join((question or "").split())
    if not question:
        return "Bạn hãy nhập một câu hỏi về Sổ tay sinh viên."

    scores = cosine_similarity(VECTORIZER.transform([question]), MATRIX).ravel()
    best_ids = scores.argsort()[::-1][:3]
    results = [
        {**DOCUMENTS[i], "score": float(scores[i])}
        for i in best_ids
        if scores[i] > 0
    ]

    if not results or results[0]["score"] < MIN_RELEVANCE:
        return (
            "Mình chưa tìm thấy nội dung đủ liên quan trong Sổ tay sinh viên. "
            "Hãy thử nêu rõ hơn từ khóa, ví dụ: học phí, đăng ký học phần hoặc ký túc xá."
        )

    answer = ["**Thông tin tìm thấy trong sổ tay:**"]
    for number, item in enumerate(results, start=1):
        answer.append(
            f"\n**{number}. {item['source']} — trang {item['page']}**\n> {item['text']}"
        )
    answer.append("\n*Vui lòng đối chiếu số trang nguồn khi cần áp dụng quy định cụ thể.*")
    return "\n".join(answer)


def chat(message, history):
    message = (message or "").strip()
    history = history or []
    if not message:
        return "", history
    return "", history + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": answer_from_handbook(message)},
    ]


CSS = """
#app {max-width: 940px; margin: auto;}
.gradio-container {font-family: system-ui, -apple-system, 'Segoe UI', sans-serif;}
#title {text-align: center; margin-bottom: 0;}
@media (max-width: 600px) {
  .gradio-container {padding: 8px !important;}
  #title {font-size: 1.35rem;}
  .message {font-size: .95rem !important;}
}
"""


with gr.Blocks(title="Chatbot Sổ tay sinh viên") as app:
    gr.Markdown("# Chatbot Sổ tay sinh viên", elem_id="title")
    gr.Markdown("Hỏi đáp dựa trên Sổ tay sinh viên; mỗi câu trả lời kèm trang nguồn.")
    chatbot = gr.Chatbot(height=480, elem_id="app")
    with gr.Row():
        question = gr.Textbox(
            placeholder="Ví dụ: Em cần làm gì để đăng ký học phần?",
            label="Câu hỏi",
            scale=8,
        )
        send = gr.Button("Gửi", variant="primary", scale=1)
    gr.Examples(
        [
            "Quy định đăng ký học phần là gì?",
            "Học phí được quy định như thế nào?",
            "Thông tin về ký túc xá",
        ],
        inputs=question,
        label="Câu hỏi gợi ý",
    )
    gr.ClearButton([question, chatbot], value="Xóa cuộc trò chuyện")
    send.click(chat, [question, chatbot], [question, chatbot])
    question.submit(chat, [question, chatbot], [question, chatbot])


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "7860"))
    app.launch(server_name="0.0.0.0", server_port=port, share=False, css=CSS)
