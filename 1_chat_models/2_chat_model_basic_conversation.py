from langchain_mistralai import ChatMistralAI
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from dotenv import load_dotenv

load_dotenv()

model = ChatMistralAI(model="mistral-large-latest")

# messages = [
#     SystemMessage(content="Solve the following mathematical problems"),
#     HumanMessage(content="What is 81 divided by 9?")
# ]
#
# result = model.invoke(messages)
# print(f"Answer from AI: {result.content}")

messages = [
    SystemMessage(content="Solve the following mathematical problems"),
    HumanMessage(content="What is 81 divided by 9?"),
    AIMessage(content="81 divided by 9 equals 9."),
    HumanMessage(content="What is 10 times 5?")
]

result = model.invoke(messages)
print(f"Answer from AI: {result.content}")