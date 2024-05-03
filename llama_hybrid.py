import os

from dotenv import load_dotenv
load_dotenv()
# os.environ["OPENAI_API_KEY"] = "sk-..."

#import tools
from trulens_eval import TruLlama, Feedback, Huggingface, Tru
from trulens_eval.schema import FeedbackResult


tru = Tru()
tru.reset_database()

##create index
from llama_index.core import (
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
)
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.node_parser import SentenceSplitter
from llama_index.llms.openai import OpenAI

splitter = SentenceSplitter(chunk_size=1024)

# load documents
documents = SimpleDirectoryReader(
    input_files=["./storage/successful APMs from CMMI.pdf"]
).load_data()

nodes = splitter.get_nodes_from_documents(documents)

# initialize storage context (by default it's in-memory)
storage_context = StorageContext.from_defaults()
storage_context.docstore.add_documents(nodes)

index = VectorStoreIndex(
    nodes=nodes,
    storage_context=storage_context,
)

## Set up retrievers
## retrieve top 10 most similar nodes using embeddings
vector_retriever = VectorIndexRetriever(index)

# retrieve the top 10 most similar nodes using bm25
bm25_retriever = BM25Retriever.from_defaults(nodes=nodes, similarity_top_k=2)

## Create hybrid retriever
from llama_index.core.retrievers import BaseRetriever

class HybridRetriever(BaseRetriever):
    def __init__(self, vector_retriever, bm25_retriever):
        self.vector_retriever = vector_retriever
        self.bm25_retriever = bm25_retriever
        super().__init__()

    def _retrieve(self, query, **kwargs):
        bm25_nodes = self.bm25_retriever.retrieve(query, **kwargs)
        vector_nodes = self.vector_retriever.retrieve(query, **kwargs)

        # combine the two lists of nodes
        all_nodes = []
        node_ids = set()
        for n in bm25_nodes + vector_nodes:
            if n.node.node_id not in node_ids:
                all_nodes.append(n)
                node_ids.add(n.node.node_id)
        return all_nodes

index.as_retriever(similarity_top_k=5)

hybrid_retriever = HybridRetriever(vector_retriever, bm25_retriever)


## Setup re-ranker
from llama_index.core.postprocessor import SentenceTransformerRerank
reranker = SentenceTransformerRerank(top_n=4, model="BAAI/bge-reranker-base")

## Setup Query Engine
from llama_index.core.query_engine import RetrieverQueryEngine
query_engine = RetrieverQueryEngine.from_args(
    retriever=hybrid_retriever,
    node_postprocessors=[reranker]
)

## First request
response = query_engine.query("List successful Alternative Payment Models?")
print(response)


###Initialize Context Relevance Checks

from trulens_eval.feedback.provider import OpenAI
from trulens_eval.schema import Select
import numpy as np

# Initialize provider class
openai = OpenAI()

bm25_context = Select.RecordCalls._retriever.bm25_retriever.retrieve.rets[:].node.text
vector_context = Select.RecordCalls._retriever.vector_retriever._retrieve.rets[:].node.text
hybrid_context = Select.RecordCalls._retriever.retrieve.rets[:].node.text
hybrid_context_filtered = Select.RecordCalls._node_postprocessors[0]._postprocess_nodes.rets[:].node.text

# Question/statement relevance between question and each context chunk.
f_context_relevance_bm25 = (
    Feedback(openai.qs_relevance, name = "BM25")
    .on_input()
    .on(bm25_context)
    .aggregate(np.mean)
    )

f_context_relevance_vector = (
    Feedback(openai.qs_relevance, name = "Vector")
    .on_input()
    .on(vector_context)
    .aggregate(np.mean)
    )

f_context_relevance_hybrid = (
    Feedback(openai.qs_relevance, name = "Hybrid")
    .on_input()
    .on(hybrid_context)
    .aggregate(np.mean)
    )

f_context_relevance_hybrid_filtered = (
    Feedback(openai.qs_relevance, name = "Hybrid Filtered")
    .on_input()
    .on(hybrid_context_filtered)
    .aggregate(np.mean)
    )

## Add Feedback
tru_recorder = TruLlama(query_engine,
    app_id='Hybrid Retriever Query Engine',
    feedbacks=[f_context_relevance_bm25,f_context_relevance_vector, f_context_relevance_hybrid, f_context_relevance_hybrid_filtered]
    )

with tru_recorder as recording:
    response = query_engine.query("Which states are successful in implementing Value Based Care?")

# with tru_recorder as recording:
#     try:
#         response = query_engine.query("Which states are successful in implementing Value Based Care?")
#     except Exception as e:
#         response = str(e)
#         print(">>>>>>>>>>>>>>>>>>>>>>Exception Occurred:", str(e))

# tru.run_dashboard()


