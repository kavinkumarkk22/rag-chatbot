import json
import faiss
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

# ---------- 1. Read the PDF and make chunks ----------
reader = PdfReader("notes.pdf")
chunks = []
chunk_size = 500
overlap = 100

for page_number, page in enumerate(reader.pages, start=1):
    text = page.extract_text()
    if not text:
        continue
    start = 0
    while start < len(text):
        chunks.append({"page": page_number, "text": text[start:start + chunk_size]})
        start += chunk_size - overlap

print("Total chunks:", len(chunks))

# ---------- 2. Create embeddings ----------
model = SentenceTransformer("all-MiniLM-L6-v2")
texts = [c["text"] for c in chunks]
embeddings = model.encode(texts, normalize_embeddings=True).astype("float32")

# ---------- 3. Put them into a FAISS index ----------
dimension = embeddings.shape[1]          # 384 numbers per chunk
index = faiss.IndexFlatIP(dimension)     # IP = inner product (same as similarity here)
index.add(embeddings)
print("Chunks stored in FAISS:", index.ntotal)

# ---------- 4. Save everything to files ----------
faiss.write_index(index, "notes.index")
with open("chunks.json", "w", encoding="utf-8") as f:
    json.dump(chunks, f, ensure_ascii=False)
print("Saved notes.index and chunks.json")

# ---------- 5. Ask questions in a loop ----------
def search(question, k=3):
    q = model.encode([question], normalize_embeddings=True).astype("float32")
    scores, ids = index.search(q, k)
    return [(chunks[i], float(s)) for i, s in zip(ids[0], scores[0])]

while True:
    question = input("\nAsk a question (or type quit): ")
    if question.strip().lower() == "quit":
        break
    for rank, (chunk, score) in enumerate(search(question), start=1):
        print(f"\n--- Match {rank} | Page {chunk['page']} | Score {score:.2f} ---")
        print(chunk["text"])