import os

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_mistralai import MistralAIEmbeddings
from langchain_mistralai import ChatMistralAI


load_dotenv()
model = "mistral-embed"

# Get the current directory
current_dir = os.path.dirname(os.path.abspath(__file__))
# Get the directory intended to use as the storage database
persistent_dir = os.path.join(current_dir, "db", "odyssey_db")

embeddings = MistralAIEmbeddings(model=model)

db = Chroma(persist_directory=persistent_dir, embedding_function=embeddings)

query = "What are the wanderings?"

retriever = db.as_retriever(search_type="similarity_score_threshold", search_kwargs={"k": 3, "score_threshold": 0.0})
relevant_docs = retriever.invoke(query)

print("\n--- Relevant Documents ---")
for i, doc in enumerate(relevant_docs, 1):
    print(f"Document {i}:\n{doc.page_content}\n")
    if doc.metadata:
        print(f"Source: {doc.metadata.get('source', 'Unknown')}\n")