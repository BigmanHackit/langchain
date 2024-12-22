from langchain.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage


# template = "Tell me all about {topic}."
# prompt_template = ChatPromptTemplate.from_template(template)
#
# user_input = input("What do you want to talk about?\n")
#
# # prompt = prompt_template.invoke({"topic": "cats"})
# prompt = prompt_template.invoke({"topic": user_input})
# print(prompt)

##########################
# template = "Tell me a short story about a {adjective} {animal}"
# prompt_template = ChatPromptTemplate.from_template(template)
#
# prompt = prompt_template.invoke({"adjective": "giant", "animal": "goat"})
# print(prompt)

##################
# messages = [
#     ("system", "You are a comedian who tells jokes about {topic}"),
#     ("human", "tell me {joke_count} jokes.")
# ]
#
# prompt_template = ChatPromptTemplate.from_messages(messages)
# prompt = prompt_template.invoke({"topic": "doctors", "joke_count": "3"})
# print(prompt)

########################
messages = [
    ("system", "You are a comedian who tells jokes about {topic}"),
    HumanMessage(content="tell me 6 jokes")
]

prompt_template = ChatPromptTemplate.from_messages(messages)
prompt = prompt_template.invoke({"topic": "doctors"})
print(prompt)