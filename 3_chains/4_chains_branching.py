from dotenv import load_dotenv
from langchain_core.runnables import RunnableLambda, RunnableBranch
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_mistralai import ChatMistralAI

load_dotenv()
model = ChatMistralAI(model="mistral-large-latest")

positive_feedback_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are an extremely helpful assistant"),
        ("human", "Write a thank you note for this positive feedback {feedback}"),
    ]
)

negative_feedback_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are an extremely helpful assistant"),
        ("human", "Write a note addressing this negative feedback {feedback}"),
    ]
)

neutral_feedback_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are an extremely helpful assistant"),
        ("human", "Write a request for more details for this neutral feedback {feedback}"),
    ]
)

escalate_feedback_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are an extremely helpful assistant"),
        ("human", "Write a note to resolve this escalated feedback {feedback}"),
    ]
)

feedback_classification_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are an extremely helpful assistant"),
        ("human", "Classify this {feedback} into positive, negative, neutral, escalate"),
    ]
)

branches = RunnableBranch(
    (
        lambda x: "positive" in x,
        positive_feedback_template | model | StrOutputParser()
    ),
    (
        lambda x: "negative" in x,
        negative_feedback_template | model | StrOutputParser()
    ),(
        lambda x: "neutral" in x,
        neutral_feedback_template | model | StrOutputParser()
    ),
    escalate_feedback_template | model | StrOutputParser()
)

classification_chain = feedback_classification_template | model | StrOutputParser()

chain = classification_chain | branches

result = chain.invoke({"feedback": "A nice product, didn't match my expectations though"})

print(result)