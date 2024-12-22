from pyexpat import features

from dotenv import load_dotenv
from langchain_core.runnables import RunnableLambda, RunnableParallel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_mistralai import ChatMistralAI

load_dotenv()
model = ChatMistralAI(model="mistral-large-latest")

prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are an expert product reviewer"),
        ("human", "List the main features of the product {product_name}"),
    ]
)


def analyze_pros(feature):
    pros_template = ChatPromptTemplate.from_messages(
        [
            ("system", "You are an expert product reviewer"),
            ("human", "Given these features: {feature}, list the pros of these features."),
        ]
    )
    return pros_template.format_prompt(feature=feature)


def analyze_cons(feature):
    cons_template = ChatPromptTemplate.from_messages(
        [
            ("system", "You are an expert product reviewer"),
            ("human", "Given these features: {feature}, list the cons of these features.")
        ]
    )
    return cons_template.format_prompt(feature=feature)


def combine_pros_and_cons(pros, cons):
    return (f"Pros: \n{pros}\n\nCons: \n{cons}")


pros_branch = (
        RunnableLambda(lambda x: analyze_pros(x)) | model | StrOutputParser()
)

cons_branch = (
        RunnableLambda(lambda x: analyze_cons(x)) | model | StrOutputParser()
)

chain = (
        prompt_template
        | model
        | StrOutputParser()
        | RunnableParallel(branches={"pros": pros_branch, "cons": cons_branch})
        | RunnableLambda(lambda x: combine_pros_and_cons(x["branches"]["pros"], x["branches"]["cons"]))
)

result = chain.invoke({"product_name": "MacBook"})

print(result)