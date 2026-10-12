from pypdf import PdfReader

# Open the PDF
reader = PdfReader("notes.pdf")

# Count the pages
print("Total pages:", len(reader.pages))

# Read the text of the first page
first_page = reader.pages[0]
text = first_page.extract_text()

print("----- Page 1 text -----")
print(text)