from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from langchain_mistralai import ChatMistralAI

load_dotenv()
model = ChatMistralAI(model="mistral-large-latest")

# template = "Tell me all about {topic}."
# prompt_template = ChatPromptTemplate.from_template(template)
#
# user_input = input("What do you want to talk about?\n")
#
# prompt = prompt_template.invoke({"topic": user_input})
# result = model.invoke(prompt)
# print(result.content)

#########################
# template = "Tell me a short story about a {adjective} {animal}"
# prompt_template = ChatPromptTemplate.from_template(template)
#
# prompt = prompt_template.invoke({"adjective": "giant", "animal": "goat"})
# result = model.invoke(prompt)
# print(result.content)

####################
# messages = [
#     ("system", "You are a comedian who tells jokes about {topic}"),
#     ("human", "tell me {joke_count} jokes.")
# ]
#
# prompt_template = ChatPromptTemplate.from_messages(messages)
# prompt = prompt_template.invoke({"topic": "doctors", "joke_count": "3"})
# result = model.invoke(prompt)
# print(result.content)

########################
messages = [
    ("system", "You are a comedian who tells jokes about {topic}"),
    HumanMessage(content="tell me 15 jokes")
]

prompt_template = ChatPromptTemplate.from_messages(messages)
user = input("Enter topic for jokes: ")
prompt = prompt_template.invoke({"topic": user})
result = model.invoke(prompt)
print(result.content)