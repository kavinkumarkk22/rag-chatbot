import os
import json
import faiss
import streamlit as st
from dotenv import load_dotenv
from google import genai
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

load_dotenv()
LLM_MODEL = "gemini-3.8-flash"
CHUNK_SIZE = 500
OVERLAP = 100

st.set_page_config(page_title="Chat with your PDF")
st.title("Chat with your PDF")
st.caption("Upload a PDF in the sidebar, then ask questions. Answers come only from the document.")


@st.cache_resource
def load_models():
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    return client, embed_model


client, embed_model = load_models()


def make_chunks(pdf_file):
    reader = PdfReader(pdf_file)
    chunks = []
    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text()
        if not text:
            continue
        start = 0
        while start < len(text):
            chunks.append({"page": page_number, "text": text[start:start + CHUNK_SIZE]})
            start += CHUNK_SIZE - OVERLAP
    return chunks


def build_index(chunks):
    texts = [c["text"] for c in chunks]
    embeddings = embed_model.encode(texts, normalize_embeddings=True).astype("float32")
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    return index


def load_default():
    with open("chunks.json", encoding="utf-8") as f:
        chunks = json.load(f)
    return {"index": faiss.read_index("notes.index"), "chunks": chunks}


# ---------- Remember things between clicks ----------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "doc" not in st.session_state:
    if os.path.exists("notes.index") and os.path.exists("chunks.json"):
        st.session_state.doc = load_default()
        st.session_state.doc_name = "notes.pdf (sample)"
    else:
        st.session_state.doc = None
        st.session_state.doc_name = None

# ---------- Sidebar: upload a PDF ----------
with st.sidebar:
    st.header("Document")
    uploaded = st.file_uploader("Upload a PDF", type="pdf")

    if uploaded is not None and st.session_state.doc_name != uploaded.name:
        with st.spinner("Reading your PDF..."):
            new_chunks = make_chunks(uploaded)
        if not new_chunks:
            st.error("No text found. This may be a scanned PDF. Try a different one.")
        else:
            with st.spinner("Indexing... (large PDFs take a minute)"):
                new_index = build_index(new_chunks)
            st.session_state.doc = {"index": new_index, "chunks": new_chunks}
            st.session_state.doc_name = uploaded.name
            st.session_state.messages = []   # start a fresh chat

    st.write("Chatting with:", st.session_state.doc_name or "nothing yet")
    if st.session_state.doc:
        st.write("Chunks:", len(st.session_state.doc["chunks"]))

if st.session_state.doc is None:
    st.info("Upload a PDF in the sidebar to begin.")
    st.stop()


# ---------- RAG functions ----------
def retrieve(question, k=4):
    doc = st.session_state.doc
    q = embed_model.encode([question], normalize_embeddings=True).astype("float32")
    k = min(k, len(doc["chunks"]))
    scores, ids = doc["index"].search(q, k)
    return [doc["chunks"][i] for i in ids[0]]


def ask(question):
    results = retrieve(question)
    context = "\n\n".join(f"[Page {c['page']}]\n{c['text']}" for c in results)

    prompt = f"""You are a helpful assistant. Answer the question using ONLY the context below.
If the answer is not in the context, reply exactly: "I couldn't find that in the document."
Keep the answer clear and short.

CONTEXT:
{context}

QUESTION: {question}

ANSWER:"""

    response = client.models.generate_content(model=LLM_MODEL, contents=prompt)
    return response.text, results


def show_sources(sources):
    with st.expander("Sources"):
        for s in sources:
            st.markdown(f"**Page {s['page']}**")
            st.write(s["text"])


# ---------- Chat ----------
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m.get("sources"):
            show_sources(m["sources"])

question = st.chat_input("Ask a question about your document...")
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching your document..."):
            try:
                answer, sources = ask(question)
            except Exception as e:
                answer, sources = f"Something went wrong: {e}", []
        st.markdown(answer)
        if sources:
            show_sources(sources)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": sources}
    )