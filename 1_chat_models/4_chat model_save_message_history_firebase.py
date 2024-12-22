from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI
from langchain_mongodb.chat_message_histories import MongoDBChatMessageHistory
from pymongo import MongoClient

load_dotenv()

model = ChatMistralAI(model="mistral-large-latest")

CONNECTION_STRING = "mongodb+srv://djhackit5:hackit131@cluster0.ujnmj.mongodb.net/langchain?retryWrites=true&w=majority"
COLLECTION_NAME = "chat_histories"
DATABASE = "langchain"

try:
    client = MongoClient(CONNECTION_STRING)
    databases = client.list_database_names()
    print("Connected to MongoDB successfully!")
    print("Available databases:", databases)
except Exception as e:
    print("Failed to connect to MongoDB:", e)

print("Initialiazing MongoDb Chat History...")
chat_history = MongoDBChatMessageHistory(
    session_id="test_session",
    connection_string=CONNECTION_STRING,
    database_name=DATABASE,
    collection_name=COLLECTION_NAME,
)

print("Chat history initialized.")
print("Current chat histories: ", chat_history.messages)

print("Start chatting with Agent0. type 'exit' to end chat.")

while True:
    human_message = input("User: ")
    if human_message == 'exit':
        break
    chat_history.add_user_message(human_message)

    ai_response = model.invoke(chat_history.messages)
    chat_history.add_ai_message(ai_response.content)

    print(f"AI: {ai_response}")