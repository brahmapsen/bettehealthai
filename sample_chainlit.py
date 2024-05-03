import chainlit as cl


@cl.action_callback("multiquery")
async def on_action(action):
    llm = cl.user_session.get("llm")
    await cl.Message(content="Multiquery").send()
    await cl.Message(content=f" {action.name} completed").send()


@cl.on_chat_start
async def on_chat_start():
  await cl.Message(content="A new chat session for Alternate Payment Model!").send()
  actions = [
        cl.Action(name="multiquery", value="example_value", description="Retrieve docs for each query!"),
    ]
  await cl.Message(content="Tools and documents loaded").send()
  llm = None
  cl.user_session.set("llm", llm)

@cl.on_message
async def on_message(msg: cl.Message):
  await cl.Message(content=msg.content).send()