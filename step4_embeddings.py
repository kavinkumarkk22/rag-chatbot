from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

# ---------- 1. Read the PDF and make chunks (same as Step 3) ----------
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

# ---------- 2. Load the embedding model ----------
# First run downloads about 90 MB, so wait for it
print("Loading model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

# ---------- 3. Turn every chunk into an embedding ----------
texts = [c["text"] for c in chunks]
embeddings = model.encode(texts, normalize_embeddings=True)

print("Embeddings shape:", embeddings.shape)
print("First 5 numbers of chunk 2:", embeddings[1][:5])

# ---------- 4. Try a search! ----------
question = "What is the difference between precision and recall?"
question_embedding = model.encode([question], normalize_embeddings=True)[0]

# Score every chunk against the question
scores = embeddings @ question_embedding

# Get the 3 best chunks
top3 = scores.argsort()[::-1][:3]

print()
print("QUESTION:", question)
for rank, i in enumerate(top3, start=1):
    print()
    print(f"--- Match {rank} | Page {chunks[i]['page']} | Score {scores[i]:.2f} ---")
    print(chunks[i]["text"])