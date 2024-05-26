# Databricks notebook source
spark.sql('GRANT USAGE ON CATALOG dbdemos TO `andrew.kraemer@databricks.com`')
spark.sql('GRANT USAGE ON DATABASE `dbdemos.rag_chatbot_andrew_kraemer_newco` TO `andrew.kraemer@databricks.com`')

# COMMAND ----------

from databricks.sdk import WorkspaceClient
import databricks.sdk.service.catalog as c

index_name = "dbdemos.rag_chatbot_andrew_kraemer_newco"

WorkspaceClient().grants.update(
    c.SecurableType.TABLE,
    index_name,
    changes=[
        c.PermissionsChange(add=[c.Privilege["SELECT"]], principal="andrew.kraemer@databricks.com")
    ],
)

# COMMAND ----------

import gradio as gr
import random
import time

DESCRIPTION = f"""
# Chatbot powered by Databricks
This chatbot helps you answers questions regarding {customer_name}. It uses retrieval augmented generation to infuse data relevant to your question into the LLM and generates an accurate response.
"""

def process_example(message: str, history: str):
    # system_prompt, max_new_tokens, temperature, top_p, top_k
    output = generate_output(message, history)
    return output

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    with gr.Row():
        gr.HTML(
            show_label=False,
            value="<img src='https://databricks.gallerycdn.vsassets.io/extensions/databricks/databricks/0.3.15/1686753455931/Microsoft.VisualStudio.Services.Icons.Default' height='40' width='40'/><div font size='1'></div>",
        )
    gr.Markdown(DESCRIPTION)
    chatbot = gr.Chatbot(height=500)
    msg = gr.Textbox(label='User Question'
                    #  , value='Ask your question'
                     )
    clear = gr.ClearButton([msg, chatbot])

    def respond(message, chat_history):
        bot_message = process_example(message, chat_history)
        chat_history.append((message, bot_message))
        time.sleep(2)
        return "", chat_history

    msg.submit(fn=respond,
        inputs=[msg, chatbot],
        outputs=[msg, chatbot])

# COMMAND ----------



# COMMAND ----------

# MAGIC %pip install gradio fastapi
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

from fastapi import FastAPI
import gradio as gr

app = FastAPI()

def greet(name):
    return "Hello demo " + name + "!"

demo = gr.Interface(fn=greet, inputs="text", outputs="text")

app = gr.mount_gradio_app(app, demo, path="/")

# COMMAND ----------

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import ChatMessage, ChatMessageRole

w = WorkspaceClient()

def respond(message, history):
    
    if len(message.strip()) == 0:
        return "ERROR the question should not be empty"

    try:
        response = w.serving_endpoints.query(
            name="databricks-dbrx-instruct", 
            messages=[ChatMessage(content=message, role=ChatMessageRole.USER)],
            temperature=1.0,
            stream=False # SDK does not support stream=True
        )
    except Exception as error:
        response = f"ERROR status_code: {type(error).__name__}"
        

    return response.choices[0].message.content

respond("what is dbrx", "asdf")

# COMMAND ----------

# MAGIC %pip install langchain==0.1.5

# COMMAND ----------

from langchain.llms import Databricks

chatbot_model_serving_endpoint = "dbdemos_rag_chatbot_andrew_kraemer_newco"
workspaceUrl = spark.conf.get("spark.databricks.workspaceUrl")


def transform_input(**request):
  full_prompt = f"""{request["prompt"]}
  Explain in bullet points."""
  request["query"] = full_prompt
  # request["stop"] = ["."]
  return request


def transform_input(**request):
  full_prompt = f"""{request["prompt"]}
  Be Concise.
  """
  request["query"] = full_prompt
  return request


def transform_output(response):
  # Extract the answer from the responses.
  return str(response)


# This model serving endpoint is created in `02_mlflow_logging_inference`
llm = Databricks(host=workspaceUrl, endpoint_name=chatbot_model_serving_endpoint, transform_input_fn=transform_input, transform_output_fn=transform_output, model_kwargs={"max_tokens": 300})

# COMMAND ----------

import requests

workspaceUrl = spark.conf.get('spark.databricks.workspaceUrl')
databricks_token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
endpoint_name = "dbdemos_rag_chatbot_andrew_kraemer_newco"

chatbot_model_serving_endpoint = f'https://{workspaceUrl}/serving-endpoints/{endpoint_name}/invocations'

message_thread = []
def reset_thread():
  global message_thread
  message_thread = []
reset_thread()



def submit_prompt(prompt):
  global message_thread 
  
  headers = {'Authorization': f'Bearer {databricks_token}', 'Content-Type': 'application/json'}


  payload = {"dataframe_split": {"columns": ["query"], "data": [[prompt]]}}

  response = requests.post(chatbot_model_serving_endpoint, headers=headers, json=payload)

  if response.status_code == 200:
      resp = response.json()
      print(resp['predictions'][0])
      bot_message = resp['predictions'][0]
      
      return bot_message
  else:
      error = response.json()
      raise ValueError(f'Error submitting job: {error}')

submit_prompt("How can I track billing usage on my workspaces?")  

# COMMAND ----------

import os
import requests
import numpy as np
import pandas as pd
import json

databricks_token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()

def create_tf_serving_json(data):
  return {'inputs': {name: data[name].tolist() for name in data.keys()} if isinstance(data, dict) else data.tolist()}

def score_model(dataset):
  url = 'https://e2-dogfood.staging.cloud.databricks.com/serving-endpoints/dbdemos_rag_chatbot_andrew_kraemer_newco/invocations'
  headers = {'Authorization': f'Bearer {databricks_token}', 'Content-Type': 'application/json'}
  ds_dict = {'dataframe_split': dataset.to_dict(orient='split')} if isinstance(dataset, pd.DataFrame) else create_tf_serving_json(dataset)
  data_json = json.dumps(ds_dict, allow_nan=True)
  print("data_json", data_json)
  response = requests.request(method='POST', headers=headers, url=url, data=data_json)
  if response.status_code != 200:
    raise Exception(f'Request failed with status {response.status_code}, {response.text}')
  return response.json()


d = {
      "query": [
        "How can I track billing usage on my workspaces?"
    ]
}
df = pd.DataFrame(d)
print(d)
score_model(df)

# COMMAND ----------

d = {
      "query": [
        "How can I track billing usage on my workspaces?"
    ]
}
df = pd.DataFrame(d)
df
ds_dict = df.to_dict(orient="split")
data_json = json.dumps(ds_dict, allow_nan=True)
score_model(d)

# COMMAND ----------

url = 'https://e2-dogfood.staging.cloud.databricks.com/serving-endpoints/dbdemos_rag_chatbot_andrew_kraemer_newco/invocations'

databricks_token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()

headers = {'Authorization': f'Bearer {databricks_token}', 'Content-Type': 'application/json'}

bort = {
  "dataframe_split": {
    "columns": [
      "query"
    ],
    "data": [
      [
        "How can I track billing usage on my workspaces?"
      ]
    ]
  }
}


data_json = json.dumps(bort, allow_nan=True)
response = requests.request(method='POST', headers=headers, url=url, data=data_json)
if response.status_code != 200:
  raise Exception(f'Request failed with status {response.status_code}, {response.text}')
response.json()

# COMMAND ----------

# MAGIC %pip install databricks-sdk --upgrade

# COMMAND ----------

dbutils.library.restartPython()


# COMMAND ----------

import itertools
# import gradio as gr
import requests
import os
# from gradio.themes.utils import sizes
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import ChatMessage, ChatMessageRole
import pandas as pd


w = WorkspaceClient()

def respond(message, history):
    
    if len(message.strip()) == 0:
        return "ERROR the question should not be empty"

    try:
        response = w.serving_endpoints.query(
            name="databricks-dbrx-instruct", 
            messages=[ChatMessage(content=message, role=ChatMessageRole.USER)],
            temperature=1.0,
            stream=False # SDK does not support stream=True
        )
    except Exception as error:
        response = f"ERROR status_code: {type(error).__name__}"
        

    return response.choices[0].message.content
respond("what is dbrx", [])

# COMMAND ----------

question = "what is dbrx?"

response = w.serving_endpoints.query(
    name="dbdemos_rag_chatbot_andrew_kraemer_newco",
    inputs=[{"query": question}]    
)

response.predictions[0]

def respond(message, history):
    
    if len(message.strip()) == 0:
        return "ERROR the question should not be empty"

    try:
        response = w.serving_endpoints.query(
            name="dbdemos_rag_chatbot_andrew_kraemer_newco",
            inputs=[{"query": message}],
              
        )
    except Exception as error:
        response = f"ERROR status_code: {type(error).__name__}"
        

    return response.predictions[0]
respond('How can I track billing usage on my workspaces?', [])

# COMMAND ----------

{
  "messages": [
    {
      "role": "user",
      "content": "Hello!"
    },
    {
      "role": "assistant",
      "content": "Hello! How can I assist you today?"
    },
    {
      "role": "user",
      "content": "What is Databricks?"
    }
  ],
  "max_tokens": 128
}

# COMMAND ----------

{
  "dataframe_split": {
    "columns": [
      "query"
    ],
    "data": [
      [
        "How can I track billing usage on my workspaces?"
      ]
    ]
  }
}