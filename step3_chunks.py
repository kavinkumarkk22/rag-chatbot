from pypdf import PdfReader

# 1. Read all pages and keep the page number with each page's text
reader = PdfReader("notes.pdf")
pages = []
for page_number, page in enumerate(reader.pages, start=1):
    text = page.extract_text()
    if text:
        pages.append({"page": page_number, "text": text})

# 2. Split each page into chunks of 500 characters
#    with a 100 character overlap, so no sentence gets cut off badly
chunk_size = 500
overlap = 100
chunks = []

for p in pages:
    text = p["text"]
    start = 0
    while start < len(text):
        piece = text[start:start + chunk_size]
        chunks.append({"page": p["page"], "text": piece})
        start += chunk_size - overlap

# 3. Show the results
print("Pages with text:", len(pages))
print("Total chunks:", len(chunks))
print("----- Chunk 1 -----")
print("Page:", chunks[0]["page"])
print(chunks[0]["text"])
print("----- Chunk 2 -----")
print("Page:", chunks[1]["page"])
print(chunks[1]["text"])