import os

from dotenv import load_dotenv
load_dotenv()

from trulens_eval import Tru
from trulens_eval.tru_custom_app import instrument
tru = Tru()
tru.reset_database()

assistant_id = os.getenv("ASSISTANT_ID")
thread_id = os.getenv("THREAD_ID")

##Create a Thread
from openai import OpenAI

class RAG_with_OpenAI_Assistant:
    def __init__(self):
        client = OpenAI()
        self.client = client

        ##Re-using assistant ID created after first RUN

        # # upload the file\
        # file = client.files.create(
        #       file=open("apm/successful APMs from CMMI.pdf", "rb"),
        #       purpose='assistants'
        # )

        # # create the assistant with access to a retrieval tool
        # assistant = client.beta.assistants.create(
        #     name="APM Assistant",
        #     instructions="You are an assistant that answers questions on Alternative Payment Model.",
        #     tools=[{"type": "retrieval"}],
        #     model="gpt-4-turbo-preview",
        #     file_ids=[file.id]
        # )
        
        # self.assistant = assistant

    @instrument
    def retrieve_and_generate(self, query: str) -> str:
        """
        Retrieve relevant text by creating and running a thread with the OpenAI assistant.
        """
        # self.thread = self.client.beta.threads.create()
        # print("Thread ID", self.thread.id)


        self.message = self.client.beta.threads.messages.create(
            thread_id=thread_id, #self.thread.id,
            role="user",
            content=query
        )

        run = self.client.beta.threads.runs.create(
            thread_id=thread_id, #self.thread.id,
            assistant_id=assistant_id, #self.assistant.id,
            instructions="Please answer any questions on Alternative Payment Model."
        )

        # Wait for the run to complete
        import time
        while run.status in ['queued', 'in_progress', 'cancelling']:
            time.sleep(1)
            run = self.client.beta.threads.runs.retrieve(
                thread_id=thread_id, #self.thread.id,
                run_id=run.id
            )

        if run.status == 'completed':
            messages = self.client.beta.threads.messages.list(
                thread_id=thread_id, #self.thread.id
            )
            response = messages.data[0].content[0].text.value
            quote = messages.data[0].content[0].text.annotations[0].file_citation.quote
        else:
            response = "Unable to retrieve information at this time."

        return response, quote
    
rag = RAG_with_OpenAI_Assistant()

##Create feedack function
from trulens_eval import Feedback, Select
from trulens_eval.feedback import Groundedness
from trulens_eval.feedback.provider.openai import OpenAI

import numpy as np

provider = OpenAI()

grounded = Groundedness(groundedness_provider=provider)

# Define a groundedness feedback function
f_groundedness = (
    Feedback(grounded.groundedness_measure_with_cot_reasons, name = "Groundedness")
    .on(Select.RecordCalls.retrieve_and_generate.rets[1])
    .on(Select.RecordCalls.retrieve_and_generate.rets[0])
    .aggregate(grounded.grounded_statements_aggregator)
)

# Question/answer relevance between overall question and answer.
f_answer_relevance = (
    Feedback(provider.relevance_with_cot_reasons, name = "Answer Relevance")
    .on(Select.RecordCalls.retrieve_and_generate.args.query)
    .on(Select.RecordCalls.retrieve_and_generate.rets[0])
)

# Question/statement relevance between question and each context chunk.
f_context_relevance = (
    Feedback(provider.qs_relevance_with_cot_reasons, name = "Context Relevance")
    .on(Select.RecordCalls.retrieve_and_generate.args.query)
    .on(Select.RecordCalls.retrieve_and_generate.rets[1])
    .aggregate(np.mean)
)

from trulens_eval import TruCustomApp
tru_rag = TruCustomApp(rag,
    app_id = 'OpenAI Assistant RAG',
    feedbacks = [f_groundedness, f_answer_relevance, f_context_relevance])

with tru_rag:
    rag.retrieve_and_generate("What is the most successful Alternative Payment Models?")

from trulens_eval import Tru

tru.get_leaderboard()
tru.run_dashboard()