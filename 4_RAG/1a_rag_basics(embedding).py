import os

from dotenv import load_dotenv
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_chroma import Chroma
from langchain_mistralai import MistralAIEmbeddings

load_dotenv()
print(load_dotenv())

# Get the current directory
current_dir = os.path.dirname(os.path.abspath(__file__))
# Get the file path down to the file itself
file_path = os.path.join(current_dir, "sources", "the_odyssey.txt")
# Get the directory intended to use as the storage database
persistent_dir = os.path.join(current_dir, "db", "odyssey_db")

# Check if intended storage database and chroma content (i.e the source for the data) exists already
if not os.path.exists(persistent_dir):
    print("Persistent directory does not exist. Initializing vector store...")

    # Check if the file path and the file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"The file {file_path} does not exist. Please check the path!"
        )

    # Extract the content with the .txt from the file path
    loader = TextLoader(file_path)
    # Load said file and store in a variable; in this case, documents
    documents = loader.load()

    # Set up the text splitter and specify chunk size and chunk overlap
    text_splitter = CharacterTextSplitter(chunk_size=3000, chunk_overlap=200)
    # With the text splitter, split the loaded documents and for better use case, store in a variable; in this case, docs
    docs = text_splitter.split_documents(documents)

    print("\n--- Document Chunks Information ---")
    print(f"Number of document chunks: {len(docs)}")
    print(f"Sample chunk:\n{docs[0].page_content}\n")

    print("\n--- Creating embeddings ---")
    # Embed the split chunks into the model
    embeddings = MistralAIEmbeddings(model="mistral-embed")
    print("\n--- Finished creating embeddings ---")

    print("\n--- Creating vector store ---")
    # Create the storage for persistence
    db = Chroma.from_documents(
        docs, embeddings, persist_directory=persistent_dir
    )
    print("\n--- Finished creating vector store ---")

# If the above steps have already been taken, no need to go through the entire process again as we already have the whole data existing in the persisted storage
else:
    print("Vector store already exists! no need to initialize.")