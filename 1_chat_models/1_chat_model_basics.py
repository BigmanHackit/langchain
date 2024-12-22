# from langchain_openai import ChatOpenAI
from langchain_mistralai import ChatMistralAI
from dotenv import load_dotenv

load_dotenv()

# model = ChatOpenAI(model="gpt-4o-mini")
model = ChatMistralAI(model="mistral-large-latest")

result = model.invoke("What is 81 divided by 9?")
print("Full result:")
print(result)
print("Content only:")
print(result.content)
