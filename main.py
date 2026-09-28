from langchain_core.messages import HumanMessage, SystemMessage
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from components.retriever import retrieve_documents

import warnings 
warnings.filterwarnings("ignore") 
from transformers import logging 
logging.set_verbosity_error()

# ---------------- LLM ----------------

llm = HuggingFacePipeline.from_model_id(
    model_id="Qwen/Qwen2.5-0.5B-Instruct",
    task="text-generation",
    pipeline_kwargs={
        "temperature": 0.7,
        "max_new_tokens": 300,
    },
)

model = ChatHuggingFace(llm=llm)
print("App starting please wait ...")

# ---------------- RAG ----------------

SYSTEM_PROMPT = """
You are a customer support assistant.

Answer the user's question using only the provided context.
If the answer is not present in the context, say:
"I don't have enough information in the provided documents.
please contact company number +91 9599048394"

Keep the answer clear and concise.
"""

def fix_message(response):
    answer = response.split("<|im_start|>assistant")[-1]
    answer = answer.split("<|im_end|>")[0].strip()
    return answer

print("----------- RAG CUSTOMER SUPPORT -----------")
print("Type 0 to exit")

while True:

    query = input("\nyou : ")
    if query == "0":
        break

    # 1. Retrieve relevant documents
    results = retrieve_documents(query, top_k=3)

    # 2. Build context
    context = ""

    for i, result in enumerate(results, start=1):
        context += f"""
    --- Document {i} ---
    Title: {result["metadata"]["title"]}
    Source: {result["metadata"]["source"]}

    {result["text"]}
    """

    # 3. Create prompt
    prompt = f"""
    Context:
    {context}

    User Question:
    {query}

    Answer based only on the context above.
    """

    # 4. Send context + question to LLM
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=prompt)
    ]

    response = model.invoke(messages)

    # 5. Clean answer
    answer = fix_message(response.content)

    print("\nBot :", answer)