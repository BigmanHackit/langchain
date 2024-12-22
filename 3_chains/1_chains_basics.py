from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser

load_dotenv()
model = ChatMistralAI(model="mistral-large-latest")

prompt_template_messages = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a comedian who tells jokes about {topic}"),
        ("human", "Tell me {joke_count} jokes")
    ]
)

chain = prompt_template_messages | model | StrOutputParser()

result = chain.invoke({"topic": "shoemaker", "joke_count": "4"})

print(result)