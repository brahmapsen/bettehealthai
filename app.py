import chainlit as cl
import os
from langchain_community.embeddings import FakeEmbeddings
from langchain_community.vectorstores import Vectara
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_openai import ChatOpenAI

from dotenv import load_dotenv

load_dotenv()

vectara_customer_id = os.getenv('VECTARA_CUSTOMER_ID')
vectara_corpus_id = os.getenv('VECTARA_CORPUS_ID')
vectara_api_key = os.getenv('VECTARA_API_KEY')

query_str = ''

def initialize_vectara():
    vectorstore = Vectara(
        vectara_customer_id=vectara_customer_id,
        vectara_corpus_id=vectara_corpus_id,
        vectara_api_key=vectara_api_key
    )
    return vectorstore

def read_pdf_vectara():
    # vectara = Vectara.from_files(["vbc/value-based-health-care.pdf"])
    vectara = Vectara.from_files(["apm/success_apm.pdf"])
    summary_config = {"is_enabled": True, "max_results": 1, "response_lang": "eng"}
    retriever = vectara.as_retriever(
        search_kwargs={"k": 1, "summary_config": summary_config}
    )
    return retriever

def get_sources(documents):
    return documents[:-1]

def get_summary(documents):
    return documents[-1].page_content

def get_summary_pdf(query_str: str):
    # (retriever | get_sources).invoke(query_str)
    retriever = cl.user_session.get("retriever")
    response =  (retriever | get_summary).invoke(query_str)
    return response

def get_summary_pdf_source(query_str: str):
    retriever = cl.user_session.get("retriever")
    response =  (retriever | get_sources).invoke(query_str)
    return response

def multiQueryReceiver(query_str: str):
    mqr = cl.user_session.get("mqr")
    return (mqr | get_summary).invoke(query_str)

def multiQueryReceiverSource(query_str: str):
    mqr = cl.user_session.get("mqr")
    return (mqr | get_sources).invoke(query_str)

# Retrieve docs for each query. Return the unique 
# union of all retrieved docs.
@cl.action_callback("multiquery")
async def on_action(action):
    print("Multiquery ")
    query_str = cl.user_session.get("query_str")
    res = multiQueryReceiver(query_str)
    await cl.Message(content=res).send()

    # src = multiQueryReceiverSource(msg.content)
    # await cl.Message(content=src).send()
    await cl.Message(content=f" {action.name} completed").send()

@cl.on_chat_start
async def on_chat_start():
    actions = [
        cl.Action(name="multiquery", value="example_value", description="Retrieve docs for each query!"),
    ]
    await cl.Message(content="Given a query, use an LLM to write a set of queries.", actions=actions).send()

    vectorstore = initialize_vectara()

    retriever = read_pdf_vectara()
    cl.user_session.set("retriever", retriever)

    #Set multiquery retriever
    llm = ChatOpenAI(temperature=0)
    mqr = MultiQueryRetriever.from_llm(retriever=retriever, llm=llm)
    cl.user_session.set("mqr", mqr)

    await cl.Message(content="A new chat session has started!").send()

@cl.on_message
async def on_message(msg: cl.Message):
    cl.user_session.set("query_str", msg.content)

    res = get_summary_pdf(msg.content)
    await cl.Message(content=res).send()

    res = get_summary_pdf_source(msg.content)
    await cl.Message(content=res).send()