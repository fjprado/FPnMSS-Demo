import re
from openai import AzureOpenAI
import os
import numpy as np
from dotenv import load_dotenv
from operator import itemgetter

load_dotenv()

client = AzureOpenAI(
  api_key = os.getenv("AZURE_OPENAI_API_KEY"),  
  api_version = "2022-12-01",
  azure_endpoint = os.getenv("AZURE_OPENAI_API_BASE")
)

#Defining helper functions
#Splits text after sentences ending in a period. Combines n sentences per chunk.
def splitter(n, f, s):
    pieces = s.split(". ")
    list_out = [{
        "id": str(i + 1), 
        "file": f, 
        "content": " ".join(pieces[i:i+n])} for i in range(0, len(pieces), n)]
    
    return list_out

# Perform light data cleaning (removing redundant whitespace and cleaning up punctuation)
def normalize_text(s, sep_token = " \n "):
    s = re.sub(r'\s+',  ' ', s).strip()
    s = re.sub(r". ,","",s)
    # remove all instances of multiple spaces
    s = s.replace("..",".")
    s = s.replace(". .",".")
    s = s.replace("\n", "")
    s = s.strip()
    
    return s

def generate_embeddings(text, model="academy-text-embedding-ada-002"): # model = "deployment_name"
    return client.embeddings.create(input=text, model=model).data[0].embedding

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def search_docs(json_list, user_query, top_n=4):
    embedding = generate_embeddings(
        user_query,
        model="academy-text-embedding-ada-002" # model should be set to the deployment name you chose when you deployed the text-embedding-ada-002 (Version 2) model
    )
    for item in json_list:
        item["similarities"] = cosine_similarity(item["embeddings"], embedding)

    res = sorted(json_list, key=itemgetter("similarities"), reverse=True)[:top_n]
    return res

def document_chunk(document, file_name):
    chunk = splitter(3, file_name, normalize_text(document))
    for item in chunk:
        item['embeddings'] = generate_embeddings(item['content'], model = 'academy-text-embedding-ada-002')
    
    return chunk

MAX_SECTION_LEN = 1000 # Set maximum token for context text

def construct_prompt(most_relevant_docs, query) -> str:    
    header = """Answer the question truthfully using context, if unsure, say "I don't know." and stop the conversation\n\nContext:\n"""
    prompt = f'{header} {"".join(doc["content"] for doc in most_relevant_docs)}\n\n Question: {query}?'
    return prompt

def summarize_text(json_list, user_query):
    prompt_i = construct_prompt(json_list, user_query) 
    print(prompt_i)

    # Using a low temperature to limit the creativity in the response. 
    response = client.completions.create(
            model= "academy-35-turbo",
            prompt = prompt_i,
            temperature = 0.0,
            max_tokens = 500,
            top_p = 1.0,
            frequency_penalty=0.5,
            presence_penalty = 0.5
        )
    print(response)
    return response.choices[0].text