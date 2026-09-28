from langchain_text_splitters import RecursiveCharacterTextSplitter

def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    chunks = []

    for document in documents:
        split_texts = splitter.split_text(document["text"])

        for i, text in enumerate(split_texts):
            chunks.append({
                "text": text,
                "title": document["title"],
                "source": document["source"],
                "chunk_id": i
            })
    return chunks