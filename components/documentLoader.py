from pathlib import Path

DATASET_PATH = Path(r"E:\projects\code\RAG_project\input")

documents = []

for (i,file_path) in enumerate(DATASET_PATH.rglob("*.txt")):
    content = file_path.read_text(encoding="utf-8", errors="ignore")
    print(i)
    if not content.strip():
        continue #skip if not found content

    # Extract title
    title = ""
    if content.startswith("Title:"):
        title = content.split("\n", 1)[0].replace("Title:", "").strip()

    # Extract text after "Text:"
    if "Text:" in content:
        text = content.split("Text:", 1)[1].strip()
    else:
        text = content.strip()

    documents.append({
        "title": title,
        "text": text,
        "source": file_path.name
    })

print(f"Loaded documents: {len(documents)}")

# Check first document
print("\nTitle:", documents[0]["title"])
print("\nSource:", documents[0]["source"])
print("\nText:", documents[0]["text"][:1000])