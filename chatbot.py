import os
import json
import faiss
import streamlit as st
from typing import List, Dict, Any, Tuple
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
import google.genai as genai
from dotenv import load_dotenv

# ======================
# CẤU HÌNH
# ======================
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")
JSON_FILE = "so_tay_sinh_vien_k65.json"
PDF_FILE = "so_tay_sinh_vien_k65.pdf"
EMBED_MODEL = "keepitreal/vietnamese-sbert"
GEMINI_MODEL = "gemini-1.5-flash"

# ======================
# HÀM ĐỌC PDF
# ======================
def read_pdf_pages(pdf_path: str, page_numbers: List[int]) -> Dict[int, str]:
    content_map = {}
    try:
        reader = PdfReader(pdf_path)
        for p in page_numbers:
            if 1 <= p <= len(reader.pages):
                page = reader.pages[p-1]
                text = page.extract_text() or ""
                content_map[p] = text.strip()
            else:
                content_map[p] = "[⚠️ Không tồn tại trang này trong PDF]"
    except Exception as e:
        st.error(f"Lỗi khi đọc PDF: {e}")
    return content_map

# ======================
# HÀM XÂY FAISS INDEX
# ======================
def load_documents(json_file: str) -> List[Dict[str, Any]]:
    with open(json_file, "r", encoding="utf-8") as f:
        return json.load(f)

def build_faiss_index(documents: List[Dict[str, Any]], model) -> Tuple[faiss.IndexFlatL2, List[Dict[str, Any]]]:
    texts = [doc["content"] for doc in documents]
    embeddings = model.encode(texts, convert_to_numpy=True)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return index, documents

def retrieve_top_k(query: str, index, documents, model, k=3) -> List[Tuple[Dict[str, Any], float]]:
    query_emb = model.encode([query], convert_to_numpy=True)
    D, I = index.search(query_emb, k)
    return [(documents[i], float(D[0][j])) for j, i in enumerate(I[0])]

# ======================
# PROMPT
# ======================
def build_prompt(question, documents: List[Tuple[Dict[str, Any], float]]):
    page_nums = [doc_meta['page'] for doc_meta, _ in documents]
    pdf_texts = read_pdf_pages(PDF_FILE, page_nums)

    context_parts = []
    for doc_meta, score in documents:
        page_num = doc_meta['page']
        content = pdf_texts.get(page_num, "[⚠️ Không tìm thấy nội dung]")
        context_parts.append(f"Nội dung trang {page_num}:\n{content}")
        
    context = "\n\n".join(context_parts)
    
    prompt = f"""
Bạn là một trợ lý ảo. Tôi sẽ đưa cho bạn một câu hỏi và một số đoạn nội dung từ file PDF.
Hãy trả lời NGẮN GỌN, TRỰC TIẾP và chỉ dựa trên nội dung đó.
Nếu không đủ thông tin, hãy nói rõ "chưa đủ dữ liệu".
Nếu KHÔNG tìm thấy, trả lời "Không tìm thấy trong tài liệu".

Câu hỏi: {question}

Các đoạn tài liệu từ PDF:
{context}

Câu trả lời:
"""
    return prompt

# ======================
# APP STREAMLIT
# ======================
def main():
    st.title("📘 Chatbot RAG với PDF + JSON + Gemini (Chat mode)")

    if not API_KEY:
        st.error("⚠️ Chưa tìm thấy GEMINI_API_KEY trong file .env")
        return

    # Init session_state
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Load model + dữ liệu 1 lần
    if "index" not in st.session_state:
        with st.spinner("Đang load dữ liệu..."):
            model = SentenceTransformer(EMBED_MODEL)
            documents = load_documents(JSON_FILE)
            index, docs = build_faiss_index(documents, model)
            st.session_state.model = model
            st.session_state.index = index
            st.session_state.docs = docs

    # Hiển thị hội thoại
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Ô chat input
    if question := st.chat_input("Nhập câu hỏi..."):
        # Lưu user message
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        # Retrieve
        retrieved_docs = retrieve_top_k(question, st.session_state.index, st.session_state.docs, st.session_state.model, k=3)

        # Build prompt
        prompt = build_prompt(question, retrieved_docs)

        # Gọi Gemini
        client = genai.Client(api_key=API_KEY)
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt
        )
        answer = response.text

        # Lưu assistant message
        st.session_state.messages.append({"role": "assistant", "content": answer})
        with st.chat_message("assistant"):
            st.markdown(answer)

        # Debug
        with st.expander("📄 Debug - Top tài liệu lấy ra"):
            for doc, score in retrieved_docs:
                st.markdown(f"**Trang {doc['page']}** (score={score:.4f})")

if __name__ == "__main__":
    main()
