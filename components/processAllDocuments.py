from documentLoader import documents
from textSplitter import split_documents
from embedding import create_embeddings
from vectorStore import store_embeddings


print("Loading documents...")
print(f"Documents loaded: {len(documents)}")


print("Splitting documents...")
chunks = split_documents(documents)

print(f"Chunks created: {len(chunks)}")


print("Creating embeddings...")
vectors = create_embeddings(chunks)

print(f"Embeddings created: {len(vectors)}")


print("Storing in ChromaDB...")
store_embeddings(chunks, vectors)


print("Document processing completed.")