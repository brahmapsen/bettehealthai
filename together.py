import os

content="Which states are successful in implementing  Alternative Payment Model (APM) \
         for Intellectual and Developmental  Disabilities(IDD) services? \
         Can you give some  recent  data like in last 2/3  years. \
         Also, what is their success story? How much was cost savings to payers?"

def get_data():
  from together import Together
  client = Together(api_key=os.environ.get("TOGETHER_API_KEY"))
  response = client.chat.completions.create(

      model="meta-llama/Llama-3-8b-chat-hf",
      messages=[{"role": "user", "content": content}],
  )
  print(response.choices[0].message.content)

if __name__ == "__main__":
    # This code will only be executed 
    # if the script is run as the main program
    print("START.........")
    get_data()