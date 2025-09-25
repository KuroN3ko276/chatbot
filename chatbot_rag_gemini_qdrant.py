import os
import streamlit as st
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Tuple, Dict, Any
import re
import json
import faiss
from dotenv import load_dotenv

# Thư viện Gemini
try:
    import google.generativeai as genai
    HAS_GEMINI = True
except Exception:
    HAS_GEMINI = False

# ---------- Load ENV ----------
load_dotenv()  # nạp biến môi trường từ file .env nếu có
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# ---------- Configuration & File Names ----------
EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2'
JSON_FILE = 'so_tay_sinh_vien_k65.json'

# ---------- Core Utilities ----------

@st.cache_resource
def get_embedding_model(name=EMBEDDING_MODEL_NAME):
    """Tải và cache mô hình nhúng."""
    return SentenceTransformer(name)

def l2_normalize_rows(x: np.ndarray) -> np.ndarray:
    """Chuẩn hóa vector về độ dài đơn vị."""
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms[norms == 0] = 1e-10
    return x / norms

def load_all_data_from_json(json_file_path: str) -> Tuple[List[Dict[str, Any]], Dict[int, str]]:
    """Tải dữ liệu từ JSON, tạo metadata cho FAISS và map nội dung trang."""
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
    except FileNotFoundError:
        st.error(f"Lỗi: Không tìm thấy file JSON: {json_file_path}. Vui lòng đảm bảo file tồn tại.")
        return [], {}
    except json.JSONDecodeError:
        st.error(f"Lỗi: File {json_file_path} bị lỗi định dạng JSON.")
        return [], {}
    
    metadata_list = []
    page_content_map = {}
    
    for item in raw_data:
        page_num = item.get("page")
        full_content = item.get("content", "").strip()
        
        if not page_num or not full_content:
            continue
            
        # Dùng dòng đầu tiên (sau khi làm sạch noise) làm tiêu đề/mô tả cho nhúng
        lines = [line.strip() for line in full_content.split('\n') if line.strip()]
        
        # Lọc bỏ các dòng noise ở đầu (ví dụ: '2 *', '4 *')
        title = ""
        for line in lines:
            if not re.match(r"^\s*\d+\s*\*|^\s*$", line):
                title = line
                break
        
        if not title:
            title = f"Nội dung trang {page_num}"
        
        if len(title) > 200:
            title = title[:150] + "..."

        metadata_list.append({
            "id": f"page_{page_num}",
            "content": title,           # Tiêu đề/mô tả để nhúng
            "page": page_num,
            "full_content": full_content # Nội dung đầy đủ
        })
        
        page_content_map[page_num] = full_content
        
    return metadata_list, page_content_map

def ensure_session_state():
    if 'faiss_index' not in st.session_state:
        st.session_state['faiss_index'] = None
    if 'messages' not in st.session_state:
        st.session_state['messages'] = []
        
    # Tải dữ liệu JSON nếu chưa có
    if 'stored_metadata' not in st.session_state or not st.session_state.get('stored_metadata', []):
        metadata_list, page_content_map = load_all_data_from_json(JSON_FILE)
        st.session_state['stored_metadata'] = metadata_list
        st.session_state['page_content_map'] = page_content_map
    
    if 'page_content_map' not in st.session_state:
        st.session_state['page_content_map'] = {}

# ---------- FAISS Utilities ----------

def upsert_metadata_faiss(metadata_list: List[Dict[str, Any]], model: SentenceTransformer):
    ensure_session_state()
    
    if not metadata_list:
        return
        
    titles = [item['content'] for item in metadata_list]
    
    embeddings = model.encode(titles, show_progress_bar=True)
    embeddings = np.asarray(embeddings).astype('float32')
    embeddings = l2_normalize_rows(embeddings)
    
    dim = embeddings.shape[1]
    
    idx = faiss.IndexFlatIP(dim)
    st.session_state['faiss_index'] = idx
    idx.add(embeddings)

def retrieve_faiss_metadata(query: str, model: SentenceTransformer, top_k: int = 5) -> List[Tuple[Dict[str, Any], float]]:
    ensure_session_state()
    idx = st.session_state.get('faiss_index', None)
    metadata = st.session_state.get('stored_metadata', [])
    
    if idx is None or idx.ntotal == 0:
        return []
        
    q_emb = model.encode([query])[0].astype('float32').reshape(1, -1)
    q_emb = l2_normalize_rows(q_emb)
    
    k = min(top_k, idx.ntotal)
    if k == 0:
        return []
    
    scores, ids = idx.search(q_emb, k)
    results = []
    for idx_id, score in zip(ids[0], scores[0]):
        if 0 <= idx_id < len(metadata):
            results.append((metadata[idx_id], float(score)))
    return results

def rerank_by_cosine_metadata(query: str, candidates: List[Dict[str, Any]], model: SentenceTransformer) -> List[Tuple[Dict[str, Any], float]]:
    if not candidates:
        return []
        
    contents = [c['content'] for c in candidates]
    q_emb = model.encode([query])[0].reshape(1, -1)
    c_embs = model.encode(contents)
    
    sims = cosine_similarity(q_emb, c_embs)[0]
    
    paired = list(zip(candidates, sims.tolist()))
    paired.sort(key=lambda x: x[1], reverse=True)
    return paired

# ---------- Prompt & Gemini Call Utilities ----------

def build_prompt(question, documents: List[Tuple[Dict[str, Any], float]]):
    context_parts = []
    
    for i, (doc_meta, score) in enumerate(documents):
        page_num = doc_meta['page']
        content = doc_meta.get('full_content', f"[LỖI: Không tìm thấy nội dung đầy đủ cho trang {page_num}]")
        context_parts.append(f"Tài liệu liên quan (Trang {page_num}):\n{content}")
        
    context = "\n\n".join(context_parts)
    
    prompt = f"""
Bạn là một trợ lý ảo. Tôi sẽ đưa cho bạn một câu hỏi và một số đoạn tài liệu (Nội dung Trang) có cấu trúc.
Hãy trả lời câu hỏi MỘT CÁCH NGẮN GỌN, TRỰC TIẾP và SÚC TÍCH, chỉ dựa trên nội dung CÓ trong các đoạn tài liệu sau.
Tuyệt đối KHÔNG bao gồm các tham chiếu đến số trang, tiêu đề hay bất kỳ ghi chú nào về tài liệu nguồn (ví dụ: 'Theo tài liệu trang 5...', 'DOC 1 nói rằng...').
Nếu thông tin không đủ để trả lời hoàn chỉnh, hãy trả lời phần có thể và nói rõ "chưa đủ dữ liệu".
Nếu KHÔNG CÓ thông tin liên quan, hãy trả lời "Không tìm thấy trong tài liệu".
Tuyệt đối KHÔNG suy diễn hoặc thêm thông tin ngoài tài liệu.

Câu hỏi: {question}

Các đoạn tài liệu:
{context}

Câu trả lời:
"""
    return prompt

def call_gemini(prompt: str, model_name: str = "gemini-1.5-flash") -> str:
    """Gọi Gemini API với API Key từ ENV."""
    api_key = GEMINI_API_KEY
         
    if not api_key:
        return "❌ Lỗi cấu hình: Không tìm thấy Gemini API Key. Vui lòng kiểm tra lại file `.env`."
    if not HAS_GEMINI:
        return "❌ Thư viện google-generativeai chưa cài. Chạy: pip install google-generativeai"
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(model_name)
        
        response = model.generate_content(prompt)
        
        if hasattr(response, 'text'):
            return response.text
        
        return "Lỗi phản hồi: Không lấy được nội dung văn bản."

    except Exception as e:
        return f"[Gemini Error] Lỗi gọi API: {e}"

# ---------- Streamlit UI & Main Logic ----------
ensure_session_state()

st.set_page_config(page_title='Chatbot RAG - Dữ liệu JSON & Tự động Tải', layout='wide')
st.title('Chatbot RAG - Sổ tay Sinh viên (Tự động tải dữ liệu JSON)')

metadata_list = st.session_state.get('stored_metadata', [])

# --- Cấu hình Sidebar ---
with st.sidebar:
    st.header('Cấu hình')
    if GEMINI_API_KEY:
        st.success('✅ Gemini API Key đã được nạp từ biến môi trường.')
    else:
        st.error('❌ Thiếu Gemini API Key. Vui lòng tạo file `.env` hoặc set biến môi trường.')
        
    top_k = st.number_input('Top-K Metadata (cho Rerank)', value=5, min_value=1, max_value=20, key='top_k_widget')
    model_choice = st.selectbox('Model Gemini (nếu dùng)', options=['gemini-1.5-flash', 'gemini-1.5-pro'], index=0)

    if st.button('Xóa FAISS & Reset Chat'):
        st.session_state['faiss_index'] = None
        st.session_state['messages'] = []
        st.rerun()

# --- 1. Tải và Nhúng Dữ liệu Tự động ---
st.subheader('1. Tải và Nhúng Dữ liệu')

if not metadata_list:
    st.error(f"⚠️ **Không tìm thấy hoặc metadata bị lỗi trong file: `{JSON_FILE}`.** Vui lòng kiểm tra lại file và cấu trúc JSON.")
else:
    st.info(f'✅ **Đã tải thành công {len(metadata_list)} mục metadata** từ file `{JSON_FILE}`.')
    st.caption('Tiêu đề (Content) của mỗi mục đã được tự động làm sạch từ dòng đầu tiên của trang.')

    if st.button('Nhúng Metadata vào FAISS', key='save_metadata_button'):
        with st.spinner('Đang nhúng TIÊU ĐỀ trang vào FAISS...'):
            try:
                model = get_embedding_model()
                upsert_metadata_faiss(metadata_list, model) 
                st.success('✅ Đã lưu Metadata (Tiêu đề) vào FAISS.')
                st.caption(f'Tổng mục Metadata hiện có: **{len(st.session_state["stored_metadata"])}**')
            except Exception as e:
                st.error(f"Lỗi khi nhúng/lưu FAISS: {e}")

# --- 2. Chatbot tương tác ---
st.subheader('2. Chatbot tương tác')
trace_expander = st.expander("🛠 Xem chi tiết Trace/Debug (Metadata, Prompt)", expanded=False)

# --- HIỂN THỊ LỊCH SỬ CHAT ---
is_faiss_ready = st.session_state['faiss_index'] is not None

if not st.session_state.messages and is_faiss_ready:
    st.session_state.messages.append({"role": "assistant", "content": "Chào bạn! Hãy hỏi tôi về Sổ tay sinh viên đã được nhúng dữ liệu."})
elif not st.session_state.messages:
    st.session_state.messages.append({"role": "assistant", "content": "Chào bạn! Vui lòng nhấn **Nhúng Metadata vào FAISS** ở phần 1 trước khi đặt câu hỏi."})

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- XỬ LÝ ĐẦU VÀO MỚI ---
prompt = st.chat_input("Hỏi về tài liệu của bạn...", key="main_chat_input")

if prompt:
    if not is_faiss_ready:
        st.error("Vui lòng nhấn **Nhúng Metadata vào FAISS** ở phần 1 trước khi đặt câu hỏi.")
    else:
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner('🤖 Đang tìm kiếm Metadata và trả lời...'):
                model = get_embedding_model()
                
                results = retrieve_faiss_metadata(prompt, model, top_k=top_k * 2) 
                
                if not results:
                    answer = "Không tìm thấy thông tin liên quan trong FAISS."
                else:
                    candidates_meta = [r[0] for r in results]
                    reranked = rerank_by_cosine_metadata(prompt, candidates_meta, model)
                    
                    retrieved_for_prompt = reranked[:top_k]
                    
                    if not retrieved_for_prompt:
                        answer = "Không tìm thấy Metadata tương tự sau khi rerank."
                    else:
                        final_prompt = build_prompt(prompt, retrieved_for_prompt)
                        
                        if GEMINI_API_KEY and HAS_GEMINI:
                            answer = call_gemini(final_prompt, model_name=model_choice)
                        else:
                            # Fallback mode
                            joined_docs = "\n\n".join([
                                f"**Mục {i+1} - {t[0]['content']} (Trang {t[0]['page']}, Sim={t[1]:.4f})**\n{t[0]['full_content']}" 
                                for i, t in enumerate(retrieved_for_prompt)
                            ])
                            answer = f"**Chế độ DEMO (Lỗi API Key)**\n\n**Tóm tắt dữ liệu liên quan:**\n{joined_docs[:2000]}...\n\n_Vui lòng kiểm tra lại API Key để có câu trả lời tự nhiên hơn._"

                    with trace_expander:
                        st.markdown('--- 📑 Top Metadata sau rerank ---')
                        for i, (meta, sim) in enumerate(retrieved_for_prompt):
                            st.markdown(f"**Mục {i+1}** — similarity (rerank)={sim:.4f}")
                            st.code(f"Tiêu đề: {meta['content']} (Trang {meta['page']})\n\nNội dung (Đầy đủ):\n{meta['full_content'][:800]}...", language='markdown')
                        
                        st.write('--- Prompt (tóm tắt, chỉ hiển thị phần đầu) ---')
                        st.code(final_prompt[:3000] + ('...' if len(final_prompt) > 3000 else ''), language='markdown')

                        st.write('🛠 Trace / Debug info', {
                            'stored_metadata_count': len(st.session_state.get('stored_metadata', [])),
                            'faiss_index_ntotal': 0 if st.session_state['faiss_index'] is None else st.session_state['faiss_index'].ntotal,
                            'retrieved_raw_count': len(results),
                            'retrieved_reranked_count': len(reranked)
                        })
                
                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
