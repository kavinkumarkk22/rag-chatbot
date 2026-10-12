import os
import json
import faiss
from dotenv import load_dotenv
from google import genai
from sentence_transformers import SentenceTransformer

# ---------- Setup ----------
load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
LLM_MODEL = "gemini-3.8-flash"

embed_model = SentenceTransformer("all-MiniLM-L6-v2")
index = faiss.read_index("notes.index")           # saved in Step 5
with open("chunks.json", encoding="utf-8") as f:  # saved in Step 5
    chunks = json.load(f)


# ---------- 1. RETRIEVE ----------
def retrieve(question, k=4):
    q = embed_model.encode([question], normalize_embeddings=True).astype("float32")
    scores, ids = index.search(q, k)
    return [chunks[i] for i in ids[0]]


# ---------- 2 + 3. AUGMENT and GENERATE ----------
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
    pages = sorted({c["page"] for c in results})
    return response.text, pages


# ---------- Chat loop ----------
print("RAG chatbot ready! Type 'quit' to stop.")
while True:
    question = input("\nYou: ")
    if question.strip().lower() == "quit":
        break
    answer, pages = ask(question)
    print("\nBot:", answer)
    print("Sources: pages", pages)