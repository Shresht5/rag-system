from langchain_huggingface import HuggingFaceEmbeddings


embeddings_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
)


def create_embeddings(chunks):
    texts = [chunk["text"] for chunk in chunks]
    print("MODEL NAME: ",embeddings_model.model_name)
    vectors = embeddings_model.embed_documents(texts)

    return vectors