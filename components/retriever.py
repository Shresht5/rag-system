from components.vectorStore import collection
from components.embedding import embeddings_model



def retrieve_documents(query, top_k=3):
    query_vector = embeddings_model.embed_query(query)
    results = collection.query(
        query_embeddings=[query_vector],
        n_results=top_k
    )
    retrieved_documents = []
    for i in range(len(results["documents"][0])):

        retrieved_documents.append({
            "text": results["documents"][0][i],
            "metadata": results["metadatas"][0][i],
            "distance": results["distances"][0][i]
        })
        
        # print(f"\n--- Result {i+1} ---")
        #  # FIXED: Added [0][i] to target the current document item
        # print("Title:", results["metadatas"][0][i].get("title", "N/A"))
        # print("Source:", results["metadatas"][0][i].get("source", "N/A"))
        # print("Distance:", results["distances"][0][i])
        # print("\nText:")
        # # FIXED: Changed "text" to the correct database key "documents"
        # print(results["documents"][0][i])
        
    return retrieved_documents