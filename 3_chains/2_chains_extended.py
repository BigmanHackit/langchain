from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from langchain_mistralai import ChatMistralAI

load_dotenv()
model = ChatMistralAI(model="mistral-large-latest")

messages = ChatPromptTemplate.from_messages(
    [("system", "You are a comedian who tells jokes about {topic}"),
    HumanMessage(content="tell {joke_count} jokes")]
)

uppercase_output = RunnableLambda(lambda x: x.upper())
word_count = RunnableLambda(lambda x: f"Word count: {len(x.split())}\n{x}")

chain = messages | model | StrOutputParser() | uppercase_output | word_count

result = chain.invoke({"topic": "hotdog", "joke_count": 6})

print(result)
