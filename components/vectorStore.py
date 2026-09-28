import chromadb
from chromadb.config import Settings

CHROMA_PATH = "./chroma_db"
client = chromadb.PersistentClient( path=CHROMA_PATH)
collection = client.get_or_create_collection( name="techqa")

def store_embeddings(chunks, vectors):
    ids = [
        f"{chunk['source']}_{chunk['chunk_id']}"
        for chunk in chunks
    ]
    documents = [
        chunk["text"]
        for chunk in chunks
    ]
    metadatas = [
        {
            "title": chunk["title"],
            "source": chunk["source"],
            "chunk_id": chunk["chunk_id"]
        }
        for chunk in chunks
    ]
    collection.add(
        ids=ids,
        documents=documents,
        embeddings=vectors,
        metadatas=metadatas
    )
    print(f"Stored {len(chunks)} chunks in ChromaDB")