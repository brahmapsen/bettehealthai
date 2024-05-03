import os

from dotenv import load_dotenv
load_dotenv()

from trulens_eval import Feedback, Tru, TruLlama
from trulens_eval.feedback import Embeddings
from trulens_eval.feedback.provider.openai import OpenAI

tru = Tru()
tru.reset_database()

from llama_index.legacy import ServiceContext

from llama_index.core import (
    SimpleDirectoryReader,
    VectorStoreIndex,
)

documents = SimpleDirectoryReader(
    input_files=["./apm/successful APMs from CMMI.pdf"]
).load_data()

from langchain.embeddings.huggingface import HuggingFaceEmbeddings

embed_model = HuggingFaceEmbeddings(model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
service_context = ServiceContext.from_defaults(embed_model = embed_model)

index = VectorStoreIndex.from_documents(documents) #, service_context = service_context)

query_engine = index.as_query_engine(top_k = 5)

## First request
response = query_engine.query("What are different successful Alternative Payment Models?")
print(response)

##
##Initialize feedback function
##

import numpy as np

# Initialize provider class
openai = OpenAI()

# Question/statement relevance between question and each context chunk.
f_qs_relevance = Feedback(openai.qs_relevance).on_input().on(
    TruLlama.select_source_nodes().node.text
    ).aggregate(np.mean)

# f_embed = Embeddings(embed_model=embed_model)

# f_embed_dist = Feedback(f_embed.cosine_distance).on_input().on(
#     TruLlama.select_source_nodes().node.text
#     ).aggregate(np.mean)

#Instrument app for logging with TruLens
tru_query_engine_recorder = TruLlama(query_engine,
    app_id='LlamaIndex_App1',
    feedbacks=[f_qs_relevance] #, f_embed_dist]
    )

with tru_query_engine_recorder as recording:
    query_engine.query("What are different successful Alternative Payment Models?")

#Explore in Dashboard
tru.run_dashboard()