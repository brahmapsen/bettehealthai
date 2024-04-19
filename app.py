import os
# import getpass
# os.environ["VECTARA_CUSTOMER_ID"] = getpass.getpass("Vectara Customer ID:")
# os.environ["VECTARA_CORPUS_ID"] = getpass.getpass("Vectara Corpus ID:")
# os.environ["VECTARA_API_KEY"] = getpass.getpass("Vectara API Key:")

# print("Read env variables ")

from dotenv import load_dotenv
load_dotenv()
vectara_customer_id = os.getenv('VECTARA_CUSTOMER_ID')
vectara_corpus_id = os.getenv('VECTARA_CORPUS_ID')
vectara_api_key = os.getenv('VECTARA_API_KEY')

from langchain_community.embeddings import FakeEmbeddings
from langchain_community.vectorstores import Vectara
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough

vectorstore = Vectara(
                vectara_customer_id=vectara_customer_id,
                vectara_corpus_id=vectara_corpus_id,
                vectara_api_key=vectara_api_key
            )

# vectara = Vectara.from_files(["vbc/value-based-health-care.pdf"])

vectara = Vectara.from_files(["apm/success_apm.pdf"])

summary_config = {"is_enabled": True, "max_results": 5, "response_lang": "eng"}
retriever = vectara.as_retriever(
    search_kwargs={"k": 3, "summary_config": summary_config}
)

def get_sources(documents):
    return documents[:-1]


def get_summary(documents):
    return documents[-1].page_content


query_str = "which states are having success in implementing Alternate Payment models. Give details."

print("Query: ", query_str)
print("")


print ( "Answer,no pre-processing:", (retriever | get_summary).invoke(query_str))
print("")
# print("Answer,no pre-processing:", str(get_summary))

# (retriever | get_sources).invoke(query_str)

from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(temperature=0)
mqr = MultiQueryRetriever.from_llm(retriever=retriever, llm=llm)

print( "Multi Query Retriever: ", (mqr | get_summary).invoke(query_str))
print("")


# (mqr | get_sources).invoke(query_str)






print("Completed.")
